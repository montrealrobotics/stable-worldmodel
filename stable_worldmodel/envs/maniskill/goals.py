"""Expert trajectory goals for any ManiSkill task, with a reusable cache."""

import contextlib
import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from .observations import single


class ExpertGoalSampler:
    """Select an expert trajectory goal and restore the planning start.

    Retry failed expert tasks or pairs with both object XY and EEF XYZ
    displacement below their minimums (meters). Verify goal replay, then
    reset and replay only the prefix preceding the selected start. Frames
    and poses include the initial state, so action i leads to frame i + 1.

    With ``cache_dir``, accepted rollouts are saved per requested seed and
    reused by later runs with the same settings, skipping expert inference;
    replay verification still runs. A file lock lets one process build each
    goal while others wait to load it.
    """

    CACHE_FIELDS = (
        'seed',
        'attempt',
        'frames',
        'poses',
        'eef_positions',
        'actions',
        'observations',
        'success',
        'start_index',
        'index',
        'object_displacement',
        'eef_displacement',
    )

    def __init__(
        self,
        goal_step_distance=25,
        max_steps=100,
        max_attempts=20,
        min_object_displacement=0.02,
        min_eef_displacement=0.02,
        cache_dir=None,
        output_dir=None,
    ):
        if goal_step_distance < 1 or max_steps < goal_step_distance:
            raise ValueError(
                'Require 1 <= goal_step_distance <= expert_max_steps'
            )
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, (int, np.integer))
            or max_attempts < 1
        ):
            raise ValueError('expert_max_attempts must be a positive integer')
        for name, threshold in (
            ('expert_min_object_displacement', min_object_displacement),
            ('expert_min_eef_displacement', min_eef_displacement),
        ):
            if not 0 <= threshold < float('inf'):
                raise ValueError(f'{name} must be nonnegative and finite')
        self.goal_step_distance = goal_step_distance
        self.max_steps = max_steps
        self.max_attempts = max_attempts
        self.min_object_displacement = min_object_displacement
        self.min_eef_displacement = min_eef_displacement
        self.cache_dir = cache_dir
        self.output_dir = output_dir

    def sample(self, env, obs, seed, options, *, start_from_beginning=True):
        """Give ``env`` an expert goal; return the planning start obs, info."""
        base = env.env.unwrapped
        requested_seed = seed
        cache_path = self.cache_path(env, seed, options, start_from_beginning)
        with self.lock(cache_path):
            cached = self.load(cache_path)
            if cached is not None:
                # Replaying a saved rollout skips the expert network passes.
                rollout = {key: cached[key] for key in self.CACHE_FIELDS}
                env.np_random.bit_generator.state = cached['rng_state']
                obs, _ = env.env.reset(seed=rollout['seed'], options=options)
                rollout['initial_state'] = base.get_state().clone()
            else:
                rollout = self.roll_out(
                    env, obs, seed, options, start_from_beginning
                )
                if cache_path is not None:
                    self.save(
                        cache_path,
                        {
                            **{key: rollout[key] for key in self.CACHE_FIELDS},
                            'rng_state': env.np_random.bit_generator.state,
                        },
                    )
        seed = rollout['seed']
        frames, poses, actions = (
            rollout['frames'],
            rollout['poses'],
            rollout['actions'],
        )
        eef_positions = rollout['eef_positions']
        start_index, index = rollout['start_index'], rollout['index']
        # Oracle replay must reconstruct the accepted scene, not the first try.
        env._evaluation_seed = seed
        if env.state_goal:
            env.goal_observation = rollout['observations'][index]
        env._goal = frames[index].copy()
        env._goal_pose = torch.as_tensor(poses[index], device=base.device)[
            None
        ]
        env._goal_eef_position = torch.as_tensor(
            eef_positions[index], device=base.device
        )[None]
        env._expert_metadata = {
            'goal_step_distance': index - start_index,
            'expert_start_step': start_index,
            'expert_length': len(actions) - start_index,
            'expert_task_success': rollout['success'],
            'expert_rollout_attempts': rollout['attempt'],
            'expert_seed': seed,
            'expert_requested_seed': requested_seed,
            'expert_start_goal_object_distance_m': rollout[
                'object_displacement'
            ],
            'expert_start_goal_eef_distance_m': rollout['eef_displacement'],
        }
        # Reset also restores controller targets and time-limit counters, which
        # BaseEnv.set_state alone does not fully restore in ManiSkill.
        obs, info = env.env.reset(seed=seed, options=options)
        torch.testing.assert_close(
            base.get_state(),
            rollout['initial_state'],
            rtol=0,
            atol=1e-6,
            msg='Expert and LEWM resets must reproduce the same start',
        )
        env.expert_actions = np.stack(actions[start_index:])
        env.expert_frames = np.stack(frames[start_index:])
        env.expert_goal_index = index - start_index
        # Replay the actual commands, including the final post-action frame.
        for action in actions[:index]:
            obs, _, _, _, _ = env.env.step(
                torch.as_tensor(action, device=base.device)[None]
            )
        replay_image = env._camera(obs)
        errors = env._goal_errors()
        env.replay_report = {
            **errors,
            'success': env._goal_reached(errors),
            'pixel_mae': float(
                np.abs(replay_image.astype(float) - env._goal).mean()
            ),
            'pixel_max_error': int(
                np.abs(replay_image.astype(int) - env._goal.astype(int)).max()
            ),
        }
        env.replay_report['passed'] = (
            env.replay_report['success']
            and env.replay_report['pixel_max_error'] <= 1
        )
        obs, info = env.env.reset(seed=seed, options=options)
        torch.testing.assert_close(
            base.get_state(), rollout['initial_state'], rtol=0, atol=1e-6
        )
        # Replay the prefix to preserve both physics and controller targets.
        env._executed_actions = [
            action.copy() for action in actions[:start_index]
        ]
        for action in env._executed_actions:
            obs, _, _, _, info = base.step(
                torch.as_tensor(action, device=base.device)[None]
            )
        # Prefix replay is setup; planning gets a fresh time-limit budget.
        if hasattr(base, '_elapsed_steps'):
            base._elapsed_steps.zero_()
        if self.output_dir:
            output = Path(self.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / f'expert_seed_{seed}.npz',
                pixels=np.stack(frames),
                action=np.stack(actions),
                obj_pose=np.stack(poses),
                eef_position=np.stack(eef_positions),
                rollout_attempts=rollout['attempt'],
                requested_seed=requested_seed,
                goal_index=index,
                start_index=start_index,
                requested_goal_step_distance=self.goal_step_distance,
                task_success=rollout['success'],
                seed=seed,
            )
            (output / f'replay_seed_{seed}.json').write_text(
                json.dumps(env.replay_report, indent=2)
            )
        if not env.replay_report['passed']:
            raise RuntimeError(
                f'Expert replay failed for seed {seed}: {env.replay_report}'
            )
        return obs, info

    def roll_out(self, env, obs, seed, options, start_from_beginning):
        """Roll out the expert, retrying until a start-goal pair is accepted."""
        base = env.env.unwrapped
        task = env.task
        # Keep retry randomness separate from start-index sampling. Consecutive
        # retry seeds would overlap the pool's seed + environment-index resets.
        retry_rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(1,))
        )
        requested_seed = seed
        attempted_seeds = {int(seed)}
        for attempt in range(1, self.max_attempts + 1):
            initial_state = base.get_state().clone()
            frames = [env._camera(obs)]
            poses = [single(task.object_pose(base))]
            eef_positions = [single(task.eef_position(base))]
            actions = []
            observations = [env._observation(obs)] if env.state_goal else []
            success = False
            for _ in range(self.max_steps):
                action = env._expert.act(obs)[0]
                obs, _, terminated, truncated, info = env.env.step(
                    torch.as_tensor(action, device=base.device)[None]
                )
                actions.append(action.copy())
                frames.append(env._camera(obs))
                poses.append(single(task.object_pose(base)))
                eef_positions.append(single(task.eef_position(base)))
                if env.state_goal:
                    observations.append(env._observation(obs))
                success = success or bool(single(info['success']))
                if bool(single(terminated)) or bool(single(truncated)):
                    break
            start_index = 0
            if not start_from_beginning:
                # Leave room for the goal offset if the rollout is long enough.
                start_index = int(
                    env.np_random.integers(
                        max(0, len(actions) - self.goal_step_distance) + 1
                    )
                )
            index = start_index + min(self.goal_step_distance, len(actions))
            object_displacement = float(
                np.linalg.norm(poses[index][:2] - poses[start_index][:2])
            )
            eef_displacement = float(
                np.linalg.norm(
                    eef_positions[index] - eef_positions[start_index]
                )
            )
            too_static = (
                object_displacement < self.min_object_displacement
                and eef_displacement < self.min_eef_displacement
            )
            if success and not too_static:
                return {
                    'seed': seed,
                    'attempt': attempt,
                    'frames': frames,
                    'poses': poses,
                    'eef_positions': eef_positions,
                    'actions': actions,
                    'observations': observations,
                    'success': success,
                    'start_index': start_index,
                    'index': index,
                    'object_displacement': object_displacement,
                    'eef_displacement': eef_displacement,
                    'initial_state': initial_state,
                }
            if attempt < self.max_attempts:
                # A deterministic expert needs a different scene to retry.
                seed = int(retry_rng.integers(0, 2**31 - 1))
                while seed in attempted_seeds:
                    seed = int(retry_rng.integers(0, 2**31 - 1))
                attempted_seeds.add(seed)
                obs, _ = env.env.reset(seed=seed, options=options)
        reason = (
            'start-goal movement too small'
            if success
            else 'expert did not solve the task'
        )
        raise RuntimeError(
            f'No valid expert rollout after {self.max_attempts} attempts '
            f'for seed {requested_seed}: {reason}'
        )

    # -- cache -------------------------------------------------------------
    def cache_path(self, env, seed, options, start_from_beginning):
        """Cache file for one expert goal, keyed by everything that shapes it.

        The GPU name is part of the key because rendered frames, and so the
        replay pixel check, can differ across GPU models.
        """
        if not self.cache_dir:
            return None
        key = json.dumps(
            {
                'version': 1,
                'seed': int(seed),
                'options': options,
                'start_from_beginning': bool(start_from_beginning),
                'env_id': env.env_id,
                'env_kwargs': env._planning_env_kwargs,
                'camera_name': env.camera_name,
                'image_size': env.image_size,
                'state_goal': env.state_goal,
                'expert_checkpoint': str(env._expert.checkpoint),
                'expert_policy_type': env._expert.policy_type,
                'goal_step_distance': self.goal_step_distance,
                'expert_max_steps': self.max_steps,
                'expert_max_attempts': self.max_attempts,
                'expert_min_object_displacement': self.min_object_displacement,
                'expert_min_eef_displacement': self.min_eef_displacement,
                'gpu': torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None,
            },
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return Path(self.cache_dir) / f'expert_{int(seed)}_{digest}.pkl'

    @staticmethod
    @contextlib.contextmanager
    def lock(path):
        """Let one process build a goal while others wait to load it."""
        if path is None:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix('.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def load(path):
        if path is None or not path.exists():
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save(path, record):
        tmp = path.with_suffix(f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as f:
            pickle.dump(record, f)
        os.replace(tmp, path)
