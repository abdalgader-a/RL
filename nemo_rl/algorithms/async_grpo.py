# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
import warnings
from typing import Any, Optional

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    MasterConfig,
    calculate_baseline_and_std_per_prompt,
    refit_policy_generation,
    validate,
)
from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import ClippedPGLossDataDict
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import Timer
from nemo_rl.utils.venvs import create_local_venv_on_each_node

TokenizerType = PreTrainedTokenizerBase


@ray.remote
class ReplayBuffer:
    """Simple replay buffer for storing trajectories."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.trajectories = []
        # Auxiliary metadata for each stored trajectory
        self.trajectory_steps = []  # collector step when generated  (for debug)
        self.trajectory_versions = []  # weight-version used for generation

    def push(self, trajectory: dict[str, Any], step: int, weight_version: int) -> None:
        """Add a trajectory with metadata.

        Args:
            trajectory: data dict
            step:       collector local step (debug only)
            weight_version: monotonic counter of the model weights used to generate
        """
        print(f"🔍 ReplayBuffer.push: Adding trajectory for step {step}")
        self.trajectories.append(trajectory)
        self.trajectory_steps.append(step)
        self.trajectory_versions.append(weight_version)

        # Remove oldest if buffer is full
        if len(self.trajectories) > self.max_size:
            removed_step = self.trajectory_steps.pop(0)
            self.trajectory_versions.pop(0)
            self.trajectories.pop(0)
            print(f"🗑️ ReplayBuffer: Removed oldest trajectory (step {removed_step})")

        print(
            f"📊 ReplayBuffer state: {len(self.trajectories)} trajectories, steps={self.trajectory_steps}"
        )

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        return {
            "total_trajectories": len(self.trajectories),
            "trajectory_steps": self.trajectory_steps,
            "trajectory_versions": self.trajectory_versions,
            "max_size": self.max_size,
        }

    def clean_old_trajectories(
        self, current_weight_version: int, max_age_steps: int
    ) -> int:
        """Remove trajectories that are too old to be useful.

        Returns:
            Number of trajectories removed
        """
        if not self.trajectories:
            return 0

        # Find trajectories to remove
        indices_to_remove = []
        for i, traj_version in enumerate(self.trajectory_versions):
            age = current_weight_version - traj_version
            if age > max_age_steps:
                indices_to_remove.append(i)

        # Remove in reverse order to maintain indices
        removed_count = 0
        for i in reversed(indices_to_remove):
            removed_step = self.trajectory_steps.pop(i)
            self.trajectory_versions.pop(i)
            self.trajectories.pop(i)
            removed_count += 1

        if removed_count > 0:
            print(f"🧹 Cleaned {removed_count} old trajectories from buffer")

        return removed_count

    def sample(
        self,
        batch_size: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[list]:
        """Sample trajectories that are not too old."""
        cleaned = self.clean_old_trajectories(current_weight_version, max_age_steps)

        if not self.trajectories:
            return None

        # Filter trajectories by age
        valid_indices = []
        total_trajectories = len(self.trajectories)

        print("🔍 ReplayBuffer sampling debug:")
        print(
            f"   current_weight_version={current_weight_version}, max_age_steps={max_age_steps}"
        )
        print(f"   trajectory_versions={self.trajectory_versions}")
        print(f"   cleaned_old_trajectories={cleaned}")

        for i, traj_version in enumerate(self.trajectory_versions):
            age = current_weight_version - traj_version

            valid = age <= max_age_steps

            print(
                (
                    f"   trajectory[{i}]: weight_version={traj_version}, age={age}, "
                    f"window={max_age_steps}, valid={valid}"
                )
            )

            if valid:
                valid_indices.append(i)

        valid_count = len(valid_indices)
        filtered_count = total_trajectories - valid_count

        if not valid_indices:
            print(
                f"⚠️  No trajectories within age limit ({max_age_steps} steps). Total: {total_trajectories}, Filtered: {filtered_count}"
            )
            return None

        if filtered_count > 0:
            print(
                f"🗑️  Filtered {filtered_count}/{total_trajectories} trajectories outside ±{max_age_steps} step window"
            )

        sampled_indices = random.sample(
            valid_indices, min(batch_size, len(valid_indices))
        )
        print(f"✅ Sampled trajectory indices: {sampled_indices}")
        return [self.trajectories[i] for i in sampled_indices]

    def size(self) -> int:
        """Return current buffer size."""
        return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        self.trajectories.clear()
        self.trajectory_steps.clear()
        self.trajectory_versions.clear()


@ray.remote
class AsyncTrajectoryCollector:
    """Collects trajectories asynchronously and adds them to replay buffer."""

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: MasterConfig,
        replay_buffer: Any,
        start_step: int = 0,
    ):
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        self.replay_buffer = replay_buffer
        self.current_step = start_step
        self.running = False

        import threading as _threading

        self._pg_lock: _threading.Lock = _threading.Lock()

        self.current_weight_version: int = start_step

        # Check if we should use async rollouts
        self._use_async_rollouts = False
        if (
            hasattr(policy_generation, "cfg")
            and "vllm_cfg" in policy_generation.cfg
            and policy_generation.cfg["vllm_cfg"].get("async_engine", False)
        ):
            self._use_async_rollouts = True
            print(
                "📦 Trajectory collector: Detected vLLM async engine; enabling async rollouts in collector"
            )

    def replace_policy_generation(
        self, new_policy_generation: GenerationInterface
    ) -> None:
        import gc

        with self._pg_lock:
            self.policy_generation = new_policy_generation
            gc.collect()

        print(
            "✅ Trajectory collector's policy generation object has been safely replaced."
        )

    def set_weight_version(self, version: int) -> None:
        """Update the local record of the model-weight version."""
        self.current_weight_version = version

    def start_collection(self, dataloader: StatefulDataLoader) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader
        print("🚀 Started continuous trajectory collection")

        import threading

        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        print("✅ Collection thread started, start_collection returning")

    def _collection_loop(self):
        """Run the collection loop in background thread."""
        try:
            for batch in self.dataloader:
                if not self.running:
                    break

                self._process_batch(batch)
                self.current_step += 1

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and add trajectories to replay buffer."""
        try:
            generation_weight_version = self.current_weight_version

            repeated_batch = batch.repeat_interleave(
                self.master_config["grpo"]["num_generations_per_prompt"]
            )

            if self._use_async_rollouts:
                with self._pg_lock:
                    final_batch, rollout_metrics = run_async_multi_turn_rollout(
                        policy_generation=self.policy_generation,
                        input_batch=repeated_batch,
                        tokenizer=self.tokenizer,
                        task_to_env=self.task_to_env,
                        max_seq_len=self.master_config["policy"][
                            "max_total_sequence_length"
                        ],
                        max_rollout_turns=self.master_config["grpo"][
                            "max_rollout_turns"
                        ],
                        greedy=False,
                    )
            else:
                # Fallback to sync rollout
                with self._pg_lock:
                    final_batch, rollout_metrics = run_multi_turn_rollout(
                        policy_generation=self.policy_generation,
                        input_batch=repeated_batch,
                        tokenizer=self.tokenizer,
                        task_to_env=self.task_to_env,
                        max_seq_len=self.master_config["policy"][
                            "max_total_sequence_length"
                        ],
                        max_rollout_turns=self.master_config["grpo"][
                            "max_rollout_turns"
                        ],
                        greedy=False,
                    )

            # Trajectory here is the complete batch required for training.
            # TODO: in future we can see if trajectory is just a prompt * num_generations_per_prompt
            trajectory = {
                "batch": final_batch,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
                "collector_step": self.current_step,
            }

            # Add to replay buffer with the weight version that was used for generation
            try:
                ray.get(
                    self.replay_buffer.push.remote(
                        trajectory, self.current_step, generation_weight_version
                    )
                )
                print(
                    f"✅ Successfully added trajectory to buffer (step {self.current_step}, weight_version {generation_weight_version})"
                )
            except Exception as e:
                print(f"❌ Failed to add trajectory to buffer: {e}")
                import traceback

                traceback.print_exc()
                return

            print(
                f"📦 Added trajectory batch (size: {final_batch.size}) to replay buffer (step {self.current_step})"
            )
            print(
                f"   Trajectory rewards: min={final_batch['total_reward'].min():.3f}, max={final_batch['total_reward'].max():.3f}, mean={final_batch['total_reward'].mean():.3f}"
            )

            try:
                buffer_size_after_push = ray.get(self.replay_buffer.size.remote())
                print(f"   Buffer size after push: {buffer_size_after_push}")
            except Exception as e:
                print(f"❌ Failed to check buffer size after push: {e}")

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()

    def get_current_step(self) -> int:
        """Return current step for debugging."""
        return self.current_step

    def get_weight_version(self) -> int:
        """Return the current weight version the collector believes it is on."""
        return self.current_weight_version

    def stop(self) -> None:
        """Stop trajectory collection."""
        self.running = False


def async_grpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
    buffer_size: int = 100,
    max_trajectory_age_steps: int = 3,
) -> None:
    """Run asynchronous GRPO training with replay buffer.

    Args:
        policy: Training policy
        policy_generation: Generation interface
        dataloader: Training data loader
        val_dataloader: Validation data loader
        tokenizer: Tokenizer
        loss_fn: Loss function
        task_to_env: Training environments
        val_task_to_env: Validation environments
        logger: Logger
        checkpointer: Checkpoint manager
        grpo_save_state: Training state
        master_config: Master configuration
        buffer_size: Maximum replay buffer size
        max_trajectory_age_steps: Maximum age (in training steps) for trajectories to be used in training
    """
    timer = Timer()
    NEED_REFIT = True

    # Setup generation interface
    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True
    assert policy_generation is not None

    # Training state
    step = grpo_save_state["step"]
    weight_version = step  # Tracks refitted weight versions
    consumed_samples = grpo_save_state["consumed_samples"]
    val_period = master_config["grpo"]["val_period"]
    val_at_start = master_config["grpo"]["val_at_start"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    # Calculate minimum buffer size from training requirements
    # Each trajectory contains (num_prompts_per_step * num_generations_per_prompt) samples
    samples_per_trajectory = (
        master_config["grpo"]["num_prompts_per_step"]
        * master_config["grpo"]["num_generations_per_prompt"]
    )
    train_gbs = master_config["policy"]["train_global_batch_size"]

    min_trajectories_needed = 1

    print("📊 Buffer requirements calculation:")
    print(f"   - num_prompts_per_step: {master_config['grpo']['num_prompts_per_step']}")
    print(
        f"   - num_generations_per_prompt: {master_config['grpo']['num_generations_per_prompt']}"
    )
    print(f"   - samples_per_trajectory: {samples_per_trajectory}")
    print(f"   - train_global_batch_size: {train_gbs}")
    print(f"   - min_trajectories_needed: {min_trajectories_needed} (async mode)")

    _replay_py_exec = get_actor_python_env("nemo_rl.algorithms.async_grpo.ReplayBuffer")
    if _replay_py_exec.startswith("uv"):
        # Lazily build a dedicated venv across all Ray nodes on-demand.
        _replay_py_exec = create_local_venv_on_each_node(
            _replay_py_exec,
            "nemo_rl.algorithms.async_grpo.ReplayBuffer",
        )

    _replay_runtime_env = {
        "py_executable": _replay_py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": _replay_py_exec,
            "UV_PROJECT_ENVIRONMENT": _replay_py_exec,
        },
    }

    replay_buffer = ReplayBuffer.options(runtime_env=_replay_runtime_env).remote(
        max_size=buffer_size
    )

    _tc_py_exec = get_actor_python_env(
        "nemo_rl.algorithms.async_grpo.AsyncTrajectoryCollector"
    )
    if _tc_py_exec.startswith("uv"):
        _tc_py_exec = create_local_venv_on_each_node(
            _tc_py_exec,
            "nemo_rl.algorithms.async_grpo.AsyncTrajectoryCollector",
        )

    _tc_runtime_env = {
        "py_executable": _tc_py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": _tc_py_exec,
            "UV_PROJECT_ENVIRONMENT": _tc_py_exec,
        },
    }

    # Initialize trajectory collector with synchronized collection
    trajectory_collector = AsyncTrajectoryCollector.options(
        runtime_env=_tc_runtime_env
    ).remote(
        policy_generation=policy_generation,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        master_config=master_config,
        replay_buffer=replay_buffer,
        start_step=step,
    )

    # Start trajectory collection in background
    collection_task = trajectory_collector.start_collection.remote(dataloader)

    # Ensure collector knows initial weight version
    trajectory_collector.set_weight_version.remote(weight_version)

    print("📦 Started continuous background trajectory collection")

    print(
        f"🚀 Starting async GRPO training with buffer_size={buffer_size}, max_age={max_trajectory_age_steps} steps"
    )

    print("⏳ Preparing policy generation for training...")
    if NEED_REFIT and POLICY_GENERATION_STALE:
        print("🔄 Refitting policy generation with actual model weights...")
        try:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            print("✅ Policy generation refit completed successfully")
            POLICY_GENERATION_STALE = False
        except Exception as e:
            print(f"❌ Policy generation refit failed: {e}")
            import traceback

            traceback.print_exc()
            return
    else:
        print("🔄 Preparing policy generation for inference...")
        try:
            policy_generation.prepare_for_generation()
            print("✅ Policy generation preparation completed successfully")
        except Exception as e:
            print(f"❌ Policy generation preparation failed: {e}")
            import traceback

            traceback.print_exc()
            return

    print("✅ Policy generation setup complete, proceeding to validation...")

    # Run validation at start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        try:
            val_metrics, validation_timings = validate(
                policy_generation,
                val_dataloader,
                tokenizer,
                val_task_to_env,
                step=0,
                master_config=master_config,
            )
            policy_generation.finish_generation()
            logger.log_metrics(val_metrics, step, prefix="validation")
            logger.log_metrics(validation_timings, step, prefix="timing/validation")
            print("✅ Initial validation completed successfully")
        except Exception as e:
            print(f"❌ Initial validation failed: {e}")
            import traceback

            traceback.print_exc()
            # Continue anyway since validation is optional

    print("✅ All setup complete, starting buffer wait...")

    # Wait for initial buffer fill
    print(
        f"⏳ Waiting for replay buffer to have sufficient trajectories (min={min_trajectories_needed})..."
    )
    wait_iterations = 0
    while True:
        buffer_size_current = ray.get(replay_buffer.size.remote())
        collector_step = ray.get(trajectory_collector.get_current_step.remote())

        print(
            f"  Wait iteration {wait_iterations}: buffer_size={buffer_size_current}/{min_trajectories_needed}, collector_step={collector_step}"
        )

        if buffer_size_current >= min_trajectories_needed:
            break

        wait_iterations += 1
        if wait_iterations > 30:
            print("🚨 TIMEOUT: Buffer never filled. Debugging buffer state...")

            buffer_debug = ray.get(replay_buffer.get_debug_info.remote())
            print(f"   Buffer debug info: {buffer_debug}")

            # Force sample to see what filtering is happening
            debug_trajectories = ray.get(
                replay_buffer.sample.remote(
                    batch_size=1,
                    current_weight_version=weight_version,
                    max_age_steps=max_trajectory_age_steps,
                )
            )
            print(f"   Debug sample result: {debug_trajectories}")

            break

        time.sleep(1.0)

    print("✅ Buffer ready! Starting training loop...")

    # Main training loop
    try:
        while step < master_config["grpo"]["max_num_steps"]:
            print(
                f"\n{'=' * 25} Step {step + 1}/{master_config['grpo']['max_num_steps']} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, step + 1)
            if policy != policy_generation:
                maybe_gpu_profile_step(policy_generation, step + 1)

            with timer.time("total_step_time"):
                # Sample trajectories from replay buffer
                print("📦 Sampling from replay buffer...")
                with timer.time("buffer_sampling"):
                    buffer_size_current = ray.get(replay_buffer.size.remote())
                    collector_step = ray.get(
                        trajectory_collector.get_current_step.remote()
                    )
                    print(
                        f"📊 Step coordination: training_step={step}, collector_step={collector_step}, max_age={max_trajectory_age_steps}, buffer_size={buffer_size_current}"
                    )

                    trajectories = ray.get(
                        replay_buffer.sample.remote(
                            batch_size=1,
                            current_weight_version=weight_version,
                            max_age_steps=max_trajectory_age_steps,
                        )
                    )

                    if trajectories is None or len(trajectories) == 0:
                        print("⏳ Buffer empty or no fresh trajectories, waiting...")

                        # Get buffer debug info to help diagnose the issue
                        buffer_debug = ray.get(replay_buffer.get_debug_info.remote())
                        buffer_size = buffer_debug["total_trajectories"]

                        if buffer_size > 0:
                            print(
                                f"🔍 Debug: Buffer has {buffer_size} trajectories but none are valid"
                            )
                            print(f"   Current weight version: {weight_version}")
                            print(f"   Max trajectory age: {max_trajectory_age_steps}")
                            print(
                                f"   Trajectory versions in buffer: {buffer_debug['trajectory_versions']}"
                            )

                        time.sleep(0.5)
                        continue

                    trajectory = trajectories[0]
                    repeated_batch = trajectory["batch"]
                    rollout_metrics = trajectory["rollout_metrics"]

                print(f"✅ Got trajectory batch (size: {repeated_batch.size})")

                print("▶ Processing rewards...")
                with timer.time("reward_calculation"):
                    prompt_only_message_logs = []
                    for message_log in repeated_batch["message_log"]:
                        prompt_only_log = []
                        for message in message_log:
                            if message["role"] == "user" or message["role"] == "system":
                                prompt_only_log.append(message)
                        prompt_only_message_logs.append(prompt_only_log)

                    prompt_batched_flat, prompt_input_lengths = (
                        batched_message_log_to_flat_message(
                            prompt_only_message_logs,
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        )
                    )
                    prompt_only_ids = prompt_batched_flat["token_ids"]

                    rewards = repeated_batch["total_reward"]

                    print("▶ Computing advantages...")

                    baseline, std = calculate_baseline_and_std_per_prompt(
                        prompt_only_ids,
                        rewards,
                        torch.ones_like(rewards),
                        leave_one_out_baseline=master_config["grpo"][
                            "use_leave_one_out_baseline"
                        ],
                    )
                    advantages = (rewards - baseline).unsqueeze(-1)

                    print(
                        f"  📊 Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}, std={rewards.std():.4f}"
                    )
                    print(
                        f"  📊 Baseline stats: min={baseline.min():.4f}, max={baseline.max():.4f}, mean={baseline.mean():.4f}"
                    )
                    print(
                        f"  📊 Advantages stats: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}, std={advantages.std():.4f}"
                    )

                    if master_config["grpo"]["normalize_rewards"]:
                        zero_std_mask = std > 0
                        advantages[zero_std_mask] = (
                            advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
                        )
                        print(
                            f"  📊 Normalized advantages stats: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}, std={advantages.std():.4f}"
                        )

                # Prepare training data (same as sync version)
                with timer.time("data_processing"):
                    # Add loss mask and advantages to each message
                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        for j, message in enumerate(message_log):
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )
                            message["advantages"] = advantages[i].expand(
                                message["token_ids"].shape
                            )

                    # Convert to flat format for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data
                    train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "advantages": flat_messages["advantages"],
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    train_data.to("cpu")

                # Training phase (same as sync version)
                print("▶ Preparing for logprob inference...")
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...")
                with timer.time("policy_and_reference_logprobs"):
                    fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                    reference_logprobs = policy.get_reference_policy_logprobs(
                        train_data
                    )["reference_logprobs"]
                    train_data["prev_logprobs"] = fprop_logprobs
                    train_data["reference_policy_logprobs"] = reference_logprobs

                print("▶ Preparing for training...")
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...")
                with timer.time("policy_training"):
                    train_results = policy.train(train_data, loss_fn)

                print("🔄 Synchronizing policy weights to trajectory collector…")
                with timer.time("weight_sync"):
                    if NEED_REFIT:
                        refit_policy_generation(
                            policy, policy_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False

                        # Notify collector about the new weight version (post-update)
                        weight_version += 1
                        trajectory_collector.set_weight_version.remote(weight_version)

                # Validation
                val_metrics, validation_timings = None, None
                is_last_step = step + 1 == master_config["grpo"]["max_num_steps"]

                if val_period > 0 and (step + 1) % val_period == 0:
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy, policy_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        policy_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=step + 1,
                        master_config=master_config,
                    )
                    policy_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, step + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(val_metrics, step + 1, prefix="validation")

                # Checkpointing (same as sync version)
                consumed_samples += master_config["grpo"]["num_prompts_per_step"]
                if master_config["checkpointing"]["enabled"] and (
                    is_last_step
                    or (step + 1) % master_config["checkpointing"]["save_period"] == 0
                ):
                    policy.prepare_for_training()

                    grpo_save_state["step"] = step + 1
                    if val_metrics is not None:
                        grpo_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in grpo_save_state:
                        del grpo_save_state["val_reward"]
                    grpo_save_state["consumed_samples"] = consumed_samples

                    if master_config["checkpointing"]["metric_name"] is not None:
                        if (
                            master_config["checkpointing"]["metric_name"]
                            not in grpo_save_state
                        ):
                            warnings.warn(
                                f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                                "Saving most recent k checkpoints instead."
                            )
                            master_config["checkpointing"]["metric_name"] = None

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {step + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            step + 1, grpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)
                    policy.offload_after_refit()

            log_data = {"content": flat_messages["content"]}
            log_data["rewards"] = rewards.tolist()
            log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
            log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")

            metrics = {
                "loss": train_results["loss"].numpy(),
                "reward": rewards.numpy(),
                "grad_norm": train_results["grad_norm"].numpy(),
            }
            metrics.update(train_results["all_mb_metrics"])
            for k, v in metrics.items():
                if k in {
                    "lr",
                    "wd",
                    "reward",
                    "global_valid_seqs",
                    "global_valid_toks",
                }:
                    metrics[k] = np.mean(v).item()
                else:
                    metrics[k] = np.sum(v).item()
            metrics.update(rollout_metrics)

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )

            # Add buffer stats
            buffer_size_current = ray.get(replay_buffer.size.remote())
            metrics["buffer_size"] = buffer_size_current

            print("\n📊 Training Results:")
            print(f"  • Loss: {metrics['loss']:.4f}")
            print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(f"  • Buffer Size: {buffer_size_current}")

            print("\n⏱️  Timing:")
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics, step + 1, prefix="train")
            logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

            timer.reset()
            step += 1

    finally:
        # Clean up
        print("🛑 Stopping trajectory collection...")
        try:
            ray.get(trajectory_collector.stop.remote())
            ray.kill(trajectory_collector)
        except Exception as e:
            print(f"⚠️ Error stopping trajectory collector: {e}")

        try:
            ray.kill(replay_buffer)
        except Exception as e:
            print(f"⚠️ Error stopping replay buffer: {e}")

        print("✅ Async GRPO training complete!")
