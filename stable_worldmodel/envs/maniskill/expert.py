"""A trained ManiSkill PPO policy that any SWM ManiSkill environment can query."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from .experts import collection_defaults
from .observations import flatten_policy_state, policy_observation


class ManiSkillExpert:
    """Load a PPO checkpoint and act on batched ManiSkill observations.

    The checkpoint's training_config.json, when present, fixes the
    environment the policy was trained in; ``check_env`` rejects mismatches
    and ``env_kwargs`` rebuilds the training cameras and reward mode, both
    before the simulator exists. ``build`` then creates the network from the
    live observation shapes on the simulator's device.
    """

    def __init__(self, checkpoint, policy_type='state', encoder='spatial_softmax'):
        if policy_type not in ('state', 'rgb'):
            raise ValueError(f'Unknown expert policy type: {policy_type}')
        self.checkpoint = checkpoint
        self.policy_type = policy_type
        self.encoder = encoder
        self.inferred, self.saved = collection_defaults(checkpoint)
        self.model = None
        self.device = None

    def check_env(self, env_id, control_mode, camera_name, image_size):
        """Fail when the evaluation env differs from the expert's training env."""
        expected = {
            'env_id': env_id,
            'control_mode': control_mode,
            'policy_type': self.policy_type,
        }
        if self.policy_type == 'rgb':
            expected.update(
                camera_name=camera_name,
                image_size=image_size,
                encoder=self.encoder,
            )
        for key, value in expected.items():
            if key in self.inferred and self.inferred[key] != value:
                raise ValueError(
                    f'Expert training config mismatch for {key}: '
                    f'{value!r} != {self.inferred[key]!r}'
                )

    def env_kwargs(self, image_size):
        """Reward mode and sensor configs matching the expert's training env."""
        if not self.saved:
            return {}, {'width': image_size, 'height': image_size}
        reward_mode = {
            'reward_mode': self.saved['environment']['kwargs']['reward_mode']
        }
        if self.policy_type != 'rgb':
            return reward_mode, {'width': image_size, 'height': image_size}
        import sapien

        # Restore camera settings used to train the RGB expert.
        sensor_configs = {}
        for name, camera in self.saved['environment']['cameras'].items():
            settings = dict(camera)
            pose = settings.pop('pose', None)
            if pose is not None:
                settings['pose'] = sapien.Pose(p=pose['p'], q=pose['q'])
            sensor_configs[name] = settings
        return reward_mode, sensor_configs

    def check_action_space(self, action_space):
        if not self.saved:
            return
        spec = self.saved['environment']
        if (
            list(action_space.shape) != spec['action_shape']
            or not np.array_equal(action_space.low, spec['action_low'])
            or not np.array_equal(action_space.high, spec['action_high'])
        ):
            raise ValueError(
                'Expert action space differs from its training config'
            )

    def build(self, env, obs, action_space):
        """Rebuild the network for env's observations and load its weights."""
        from . import experts

        self.device = env.unwrapped.device
        self.action_space = action_space
        if self.policy_type == 'rgb':
            options = (
                {
                    key: self.saved['agent'][key]
                    for key in ('std_mode', 'initial_std', 'min_std', 'max_std')
                }
                if self.saved
                else {}
            )
            model = experts.RGBExpert(
                env,
                policy_observation(obs, self.device, 'rgb'),
                encoder=self.encoder,
                **options,
            )
        else:
            state = flatten_policy_state(obs, self.device)
            spaces = SimpleNamespace(
                single_observation_space=gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=tuple(state.shape[1:]),
                    dtype=np.float32,
                ),
                single_action_space=action_space,
            )
            model = experts.StateExpert(spaces)
        model = model.to(self.device)
        model.load_state_dict(
            torch.load(
                self.checkpoint, map_location=self.device, weights_only=True
            )
        )
        model.eval()
        self.model = model

    @torch.inference_mode()
    def act(self, obs):
        """Deterministic actions for every env, clipped, as a NumPy array."""
        action = self.model.get_action(
            policy_observation(obs, self.device, self.policy_type),
            deterministic=True,
        )
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.clip(
            np.asarray(action), self.action_space.low, self.action_space.high
        )
        if not np.isfinite(action).all():
            raise ValueError('Expert produced nonfinite action')
        return action
