"""Rollout alignment and Lance serialization checks without a simulator."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    'collect_maniskill_pusht',
    Path(__file__).parents[2] / 'scripts/data/collect_maniskill_pusht.py',
)
_collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_collect)
collect_episode = _collect.collect_episode
collect_episode_batch = _collect.collect_episode_batch
named_observation_leaves = _collect.named_observation_leaves
episode_modes = _collect.episode_modes
policy_observation = _collect.policy_observation


class FakeEnv:
    unwrapped = property(lambda self: self)

    def __init__(self, end_kind):
        self.end_kind = end_kind
        self.actions = []

    def reset(self, seed):
        self.seed = seed
        self.t = 0
        return self.observation(), {}

    def observation(self):
        return {
            'agent': {'qpos': torch.tensor([[float(self.t)]])},
            'extra': {'nested': {'goal': torch.tensor([[7.0, 8.0]])}},
            'sensor_data': {
                'base_camera': {
                    'rgb': torch.full((1, 2, 2, 3), self.t, dtype=torch.uint8)
                },
                'other_camera': {
                    'rgb': torch.full((1, 2, 2, 3), 99, dtype=torch.uint8)
                },
            },
        }

    def get_state(self):
        # A stand-in for ManiSkill's batched, flat, restorable simulator state.
        return torch.tensor([[float(self.t), float(self.t + 100)]])

    def render_rgb_array(self, *args, **kwargs):
        raise AssertionError(
            'Collector must use sensor observations, not render()'
        )

    def step(self, action):
        self.actions.append(action.clone())
        self.t += 1
        return (
            self.observation(),
            torch.tensor([1.0]),
            torch.tensor(
                [
                    (self.t == 3 and self.end_kind == 'terminated')
                    or (self.t == 2 and self.end_kind == 'success')
                ]
            ),
            torch.tensor([self.t == 3 and self.end_kind == 'truncated']),
            {
                'success': torch.tensor(
                    [self.t == 2 and self.end_kind == 'success']
                )
            },
        )


class FakeAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.zeros(1))
        self.deterministic_calls = []

    def get_action(self, obs, deterministic):
        self.deterministic_calls.append(deterministic)
        # Check the exact agent-then-extra ordering supplied to the policy.
        assert tuple(obs.shape) == (1, 3)
        assert torch.equal(obs[:, 1:], torch.tensor([[7.0, 8.0]]))
        return obs[:, :1] + 10


class CollectionTests(unittest.TestCase):
    def test_rgb_observation_matches_training_wrapper(self):
        from types import SimpleNamespace

        from mani_skill.utils.wrappers.flatten import (
            FlattenRGBDObservationWrapper,
        )

        obs = FakeEnv('horizon').reset(seed=0)[0]
        obs['sensor_param'] = {}
        wrapper = SimpleNamespace(
            include_rgb=True,
            include_depth=False,
            include_state=False,
            base_env=SimpleNamespace(device='cpu'),
        )
        expected = FlattenRGBDObservationWrapper.observation(
            wrapper, dict(obs)
        )
        actual = policy_observation(obs, 'cpu', 'rgb')
        self.assertEqual(set(actual), {'rgb'})
        torch.testing.assert_close(actual['rgb'], expected['rgb'])
        self.assertIn('sensor_data', obs)
        self.assertEqual(len(named_observation_leaves(obs)), 2)

    def test_rgb_xy_rollout_and_clipping(self):
        from types import SimpleNamespace

        class RGBEnv(FakeEnv):
            single_action_space = SimpleNamespace(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])
            )

        class RGBAgent(FakeAgent):
            def get_action(self, obs, deterministic):
                assert set(obs) == {'rgb'}
                assert obs['rgb'].shape == (1, 2, 2, 6)
                assert deterministic
                return torch.tensor([[2.0, -2.0]])

        env = RGBEnv('terminated')
        episode, _, _ = collect_episode(
            env,
            RGBAgent(),
            seed=3,
            max_steps=8,
            camera_name='base_camera',
            stochastic=False,
            policy_type='rgb',
        )
        self.assertEqual(len(episode['action']), 3)
        np.testing.assert_array_equal(episode['action'], [[1.0, -1.0]] * 3)
        for step, action in enumerate(env.actions):
            np.testing.assert_array_equal(action[0], episode['action'][step])
            self.assertEqual(episode['pixels'][step][0, 0, 0], step)
            self.assertEqual(episode['obs_agent_qpos'][step][0], step)

    def test_rgb_expert_numeric_only_dataset(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        import lance

        from stable_worldmodel.data import EPISODE_DATA_KEY, LanceWriter

        class RGBEnv(FakeEnv):
            single_action_space = SimpleNamespace(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])
            )

        class RGBAgent(FakeAgent):
            def get_action(self, obs, deterministic):
                # Images still reach the expert, including the current step.
                assert obs['rgb'].shape == (1, 2, 2, 6)
                value = obs['rgb'][0, 0, 0, 0].float()
                return torch.stack([value / 10, -value / 10])[None]

        episodes = []
        for save_rgb in (True, False):
            env = RGBEnv('terminated')
            episode, _, _ = collect_episode(
                env,
                RGBAgent(),
                seed=3,
                max_steps=8,
                camera_name='base_camera',
                stochastic=False,
                policy_type='rgb',
                save_rgb=save_rgb,
            )
            np.testing.assert_array_equal(
                episode['action'], torch.cat(env.actions).numpy()
            )
            episodes.append(episode)
        self.assertNotIn('pixels', episodes[1])
        for key in episodes[1]:
            np.testing.assert_array_equal(episodes[0][key], episodes[1][key])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.lance'
            episodes[1][EPISODE_DATA_KEY] = {
                'collection_mode': 'deterministic'
            }
            with LanceWriter(path, mode='error') as writer:
                writer.write_episodes([episodes[1]])
            table = lance.dataset(str(path)).to_table()
            self.assertNotIn('pixels', table.column_names)
            self.assertNotIn('collection_mode', table.column_names)
            metadata = lance.dataset(
                str(path.with_name('state_episodes.lance'))
            ).to_table()
            self.assertEqual(
                metadata['collection_mode'].to_pylist(), ['deterministic']
            )
            self.assertEqual(table['step_idx'].to_pylist(), [0, 1, 2])
            np.testing.assert_allclose(
                table['action'].to_pylist(), episodes[1]['action']
            )
            self.assertEqual(
                table['obs_agent_qpos'].to_pylist(), [[0.0], [1.0], [2.0]]
            )

    def test_spatial_softmax_xy_policy(self):
        from types import SimpleNamespace

        import gymnasium as gym

        from stable_worldmodel.envs.maniskill.experts import RGBExpert as Agent

        torch.set_num_threads(1)
        env = SimpleNamespace(
            unwrapped=SimpleNamespace(
                single_action_space=gym.spaces.Box(-1.0, 1.0, (2,))
            )
        )
        obs = {'rgb': torch.zeros(1, 128, 128, 3, dtype=torch.uint8)}
        agent = Agent(env, obs, encoder='spatial_softmax').eval()
        restored = Agent(env, obs, encoder='spatial_softmax').eval()
        restored.load_state_dict(agent.state_dict())
        action = restored.get_action(obs, deterministic=True)
        self.assertEqual(action.shape, (1, 2))
        torch.testing.assert_close(
            action, agent.get_action(obs, deterministic=True)
        )

    def test_mixture_counts_and_seed(self):
        from collections import Counter

        modes = episode_modes(10, 'mixed', (0.5, 0.3, 0.2), 42)
        self.assertEqual(
            Counter(modes), {'deterministic': 5, 'stochastic': 3, 'noisy': 2}
        )
        self.assertEqual(modes, episode_modes(10, 'mixed', (5, 3, 2), 42))
        self.assertEqual(len(episode_modes(7, 'mixed', (1, 1, 1), 0)), 7)

    def test_noisy_actions_are_reproducible_bounded_and_recorded(self):
        from types import SimpleNamespace

        class ZeroAgent(FakeAgent):
            def get_action(self, obs, deterministic):
                self.assert_deterministic = deterministic
                return torch.zeros((1, 1))

        def rollout(seed):
            env, agent = FakeEnv('horizon'), ZeroAgent()
            env.single_action_space = SimpleNamespace(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
            )
            episode, _, _ = collect_episode(
                env,
                agent,
                seed=seed,
                max_steps=6,
                camera_name='base_camera',
                stochastic=False,
                noise_std=0.2,
                noise_hold_steps=2,
            )
            actions = np.asarray(episode['action'])
            np.testing.assert_array_equal(
                actions, torch.cat(env.actions).numpy()
            )
            self.assertTrue(agent.assert_deterministic)
            self.assertTrue((np.abs(actions) <= 1).all())
            np.testing.assert_array_equal(actions[::2], actions[1::2])
            return actions

        first = rollout(7)
        np.testing.assert_array_equal(first, rollout(7))
        self.assertFalse(np.array_equal(first, rollout(8)))

    def test_alignment_and_episode_boundaries(self):
        for end_kind in ('terminated', 'truncated', 'horizon'):
            with self.subTest(end_kind=end_kind):
                env, agent = FakeEnv(end_kind), FakeAgent()
                episode, total_return, success = collect_episode(
                    env,
                    agent,
                    seed=42,
                    max_steps=3 if end_kind == 'horizon' else 10,
                    camera_name='base_camera',
                    stochastic=False,
                )
                np.testing.assert_array_equal(
                    np.asarray(episode['pixels'])[:, 0, 0, 0], [0, 1, 2]
                )
                np.testing.assert_array_equal(
                    episode['state'], [[0, 100], [1, 101], [2, 102]]
                )
                np.testing.assert_array_equal(
                    np.asarray(episode['obs_agent_qpos'])[:, 0], [0, 1, 2]
                )
                np.testing.assert_array_equal(
                    episode['obs_extra_nested_goal'], [[7, 8]] * 3
                )
                self.assertNotIn('policy_state', episode)
                np.testing.assert_array_equal(
                    np.asarray(episode['action'])[:, 0], [10, 11, 12]
                )
                np.testing.assert_array_equal(
                    episode['action'], torch.cat(env.actions).numpy()
                )
                self.assertEqual(env.seed, 42)
                self.assertEqual(total_return, 3)
                self.assertFalse(success)
                self.assertEqual(agent.deterministic_calls, [True] * 3)

    def test_success_termination_stops_collection(self):
        env, agent = FakeEnv('success'), FakeAgent()
        episode, total_return, success = collect_episode(
            env,
            agent,
            seed=0,
            max_steps=100,
            camera_name='base_camera',
            stochastic=False,
        )
        self.assertTrue(success)
        self.assertEqual(total_return, 2)
        self.assertEqual(len(env.actions), 2)
        self.assertEqual(agent.deterministic_calls, [True, True])
        self.assertTrue(all(len(values) == 2 for values in episode.values()))
        np.testing.assert_array_equal(episode['action'], [[10], [11]])
        np.testing.assert_array_equal(episode['obs_agent_qpos'], [[0], [1]])

    def test_stochastic_mode(self):
        agent = FakeAgent()
        collect_episode(
            FakeEnv('horizon'),
            agent,
            seed=0,
            max_steps=1,
            camera_name='base_camera',
            stochastic=True,
        )
        self.assertEqual(agent.deterministic_calls, [False])

    def test_sensor_selection(self):
        episode, _, _ = collect_episode(
            FakeEnv('horizon'),
            FakeAgent(),
            seed=0,
            max_steps=1,
            camera_name='other_camera',
            stochastic=False,
        )
        self.assertTrue((np.asarray(episode['pixels']) == 99).all())

    def test_lance_serialization(self):
        import tempfile
        from pathlib import Path

        import pyarrow as pa

        from stable_worldmodel.data import LanceWriter

        episode, _, _ = collect_episode(
            FakeEnv('horizon'),
            FakeAgent(),
            seed=0,
            max_steps=3,
            camera_name='base_camera',
            stochastic=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.lance'
            writer = LanceWriter(path, mode='error')
            # Exercise the actual SWM schema inference that rejected stacked
            # arrays, plus image encoding and numeric-column serialization.
            # No database connection is needed for this regression check.
            writer._init_schema(episode)
            table = pa.Table.from_batches(
                [
                    writer._batch_from_episode(episode, ep_extra={})
                    for _ in range(3)
                ]
            )
            self.assertEqual(table.num_rows, 9)
            self.assertEqual(
                table['episode_idx'].to_pylist(), [0] * 3 + [1] * 3 + [2] * 3
            )
            np.testing.assert_array_equal(
                table['action'].to_pylist(), [[10], [11], [12]] * 3
            )
            np.testing.assert_array_equal(
                table['state'].to_pylist(), [[0, 100], [1, 101], [2, 102]] * 3
            )
            np.testing.assert_array_equal(
                table['obs_agent_qpos'].to_pylist(), [[0], [1], [2]] * 3
            )
            np.testing.assert_array_equal(
                table['obs_extra_nested_goal'].to_pylist(), [[7, 8]] * 9
            )
            self.assertTrue(all(table['pixels'].to_pylist()))

    def test_unknown_sensor(self):
        with self.assertRaisesRegex(ValueError, 'Unknown sensor'):
            collect_episode(
                FakeEnv('horizon'),
                FakeAgent(),
                seed=0,
                max_steps=1,
                camera_name='missing',
                stochastic=False,
            )

    def test_noisy_replays_follow_expert_actions_from_same_reset(self):
        from types import SimpleNamespace

        class BatchEnv:
            unwrapped = property(lambda self: self)
            num_envs = 2
            single_action_space = SimpleNamespace(
                shape=(1,),
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
            )

            def reset(self, seed):
                self.x = torch.tensor(seed, dtype=torch.float32).reshape(2, 1)
                return self.observation(), {}

            def observation(self):
                return {'agent': {'qpos': self.x.clone()}, 'extra': {}}

            def get_state(self):
                return self.x.clone()

            def step(self, action):
                self.x = self.x + action
                done = torch.zeros(2, dtype=torch.bool)
                return (
                    self.observation(),
                    torch.zeros(2),
                    done,
                    done,
                    {'success': done},
                )

        class StepAgent(FakeAgent):
            def get_action(self, obs, deterministic):
                return torch.full((len(obs), 1), 0.5)

        def run(**kwargs):
            return collect_episode_batch(
                BatchEnv(),
                StepAgent(),
                seeds=[3, 5],
                max_steps=4,
                camera_name=None,
                noise_hold_steps=1,
                save_rgb=False,
                **kwargs,
            )

        expert = run(modes=['deterministic'] * 2)
        actions = torch.full((4, 2, 1), 0.5)
        replay = dict(
            modes=['replay'] * 2, replay_actions=actions, replay_lengths=[4, 2]
        )
        exact = run(noise_std=0.0, **replay)
        for (expert_episode, _, _), (replay_episode, _, _) in zip(
            expert, exact
        ):
            np.testing.assert_array_equal(
                np.asarray(replay_episode['state']),
                np.asarray(expert_episode['state'])[
                    : len(replay_episode['state'])
                ],
            )
        self.assertEqual(
            [len(episode['action']) for episode, _, _ in exact], [4, 2]
        )
        noisy = run(noise_std=0.5, noise_seed=11, **replay)
        noisy_actions = np.asarray(noisy[0][0]['action'])
        self.assertFalse(np.allclose(noisy_actions, 0.5))
        self.assertTrue((np.abs(noisy_actions) <= 1).all())
        np.testing.assert_array_equal(
            noisy_actions,
            np.asarray(
                run(noise_std=0.5, noise_seed=11, **replay)[0][0]['action']
            ),
        )
        self.assertFalse(
            np.array_equal(
                noisy_actions,
                np.asarray(
                    run(noise_std=0.5, noise_seed=12, **replay)[0][0]['action']
                ),
            )
        )

    def test_state_key_collision(self):
        with self.assertRaisesRegex(ValueError, 'collide'):
            named_observation_leaves(
                {'agent': {'a_b': 1, 'a': {'b': 2}}, 'extra': {}}
            )


def test_assemble_correct_chunks_uses_accepted_slots_and_falls_back():
    groups, n, chunk, length = 2, 4, 2, 5  # chunks [0,2) [2,4) [4,5) partial
    action = np.arange(length)[:, None] * 100 + np.arange(groups * n)[None]
    arrays = {'action': action[..., None].astype(np.float32)}
    accepted = [np.ones((groups, n), bool) for _ in range(3)]
    accepted[1][0] = [True, False, False, True]  # group 0 chunk 1: only slot 3
    out = _collect.assemble_correct_chunk_episodes(
        arrays,
        [length, 3],
        accepted,
        candidates=n,
        num_alternatives=2,
        chunk=chunk,
        rng=np.random.default_rng(0),
    )
    expert, first, second = out[0]
    np.testing.assert_array_equal(expert[1], [0] * length)
    for _, slots, _ in (first, second):
        assert slots[0] == slots[1] and slots[0] in (1, 2, 3)
        assert slots[4] == 0  # partial last chunk keeps the expert
    assert {first[1][0], second[1][0]} <= {1, 2, 3}
    assert first[1][0] != second[1][0]  # distinct samples per chunk
    np.testing.assert_array_equal(first[1][2:4], [3, 3])
    np.testing.assert_array_equal(second[1][2:4], [0, 0])
    assert (first[2], second[2]) == (1, 2)
    assert [len(e[0]['action']) for e in out[1]] == [3, 3, 3]
    assert (
        out[1][1][1][0] >= n and out[1][1][2] == 1
    )  # group 1 slots; partial chunk
    np.testing.assert_array_equal(
        np.stack(first[0]['action'])[:, 0], action[np.arange(length), first[1]]
    )


def test_correct_chunk_batch_records_expert_and_alternatives(monkeypatch):
    from types import SimpleNamespace

    import stable_worldmodel.envs.maniskill.chunks as chunks

    groups, n, terminate = 2, 3, 4

    class ChunkEnv:
        unwrapped = property(lambda self: self)
        num_envs = groups * n
        single_action_space = SimpleNamespace(
            shape=(2,), low=-np.ones(2), high=np.ones(2)
        )

        def reset(self, seed):
            assert seed == [10] * n + [11] * n
            self.t = 0
            return self.observation(), {}

        def observation(self):
            return {
                'agent': {
                    'qpos': torch.full((self.num_envs, 1), float(self.t))
                },
                'extra': {},
                'sensor_data': {
                    'base_camera': {
                        'rgb': torch.full(
                            (self.num_envs, 2, 2, 3), self.t, dtype=torch.uint8
                        )
                    }
                },
            }

        def get_state(self):
            return torch.full((self.num_envs, 2), float(self.t))

        def step(self, action):
            self.t += 1
            done = torch.full((self.num_envs,), self.t >= terminate)
            return (
                self.observation(),
                torch.ones(self.num_envs),
                done,
                torch.zeros(self.num_envs, dtype=torch.bool),
                {'success': done},
            )

    class Agent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def get_logstd(self):
            return torch.zeros(1, 2)

        def get_action(self, obs, deterministic):
            assert deterministic
            return torch.zeros(len(obs), 2)

    env = ChunkEnv()
    mask = torch.tensor([True, True, False] * groups)
    monkeypatch.setattr(
        chunks, 'restore_from_source', lambda e, s: e.observation()
    )
    monkeypatch.setattr(chunks, 'snapshot', lambda e: None)
    monkeypatch.setattr(chunks, 'end_errors', lambda end, s: None)
    monkeypatch.setattr(chunks, 'accepted', lambda errors, tol: mask)
    out = _collect.collect_correct_chunk_batch(
        env,
        Agent(),
        seeds=[10, 11],
        candidates=n,
        num_alternatives=2,
        chunk=2,
        max_steps=10,
        camera_name='base_camera',
        std_scales=(1.0,),
        tolerance=None,
        noise_seed=0,
    )
    assert len(out) == groups
    for rollouts in out:
        modes = [(r[0], r[1]) for r in rollouts]
        assert modes == [
            ('deterministic', -1),
            ('correct_chunks', 0),
            ('correct_chunks', 1),
        ]
        expert, alt, missing = (r[2] for r in rollouts)
        assert len(expert['action']) == len(alt['action']) == terminate
        assert all((a == 0).all() for a in expert['action'])
        assert all((a != 0).any() for a in alt['action'])  # slot 1 noise
        assert [r[5] for r in rollouts] == [0, 0, 2]  # only one accepted
        assert all((a == 0).all() for a in missing['action'])
        assert rollouts[0][3] == terminate and rollouts[0][4]
        assert len(expert['pixels']) == terminate


if __name__ == '__main__':
    unittest.main()


def test_select_alternatives_keeps_successful_slots_in_order():
    def episode(start, tag):
        return {'state': [np.full(2, start, np.float32)], 'action': [tag]}

    successes = [True, False, True, True, True, False, False, True]
    batch = [
        (episode(slot // 4, slot), float(slot), success)
        for slot, success in enumerate(successes)
    ]
    groups, similar = _collect.select_alternatives(batch, 4, 2)
    assert similar == 0
    assert [[(r[0], r[1], r[2]['action'][0]) for r in g] for g in groups] == [
        [
            ('deterministic', -1, 0),
            ('alternative', 0, 2),
            ('alternative', 1, 3),
        ],
        [('deterministic', -1, 4), ('alternative', 0, 7)],
    ]
    batch[1] = (episode(9, 1), 1.0, False)
    with pytest.raises(ValueError, match='initial state'):
        _collect.select_alternatives(batch, 4, 2)


def test_select_alternatives_drops_paths_too_close_to_kept_ones():
    def episode(offset, tag, length=10):
        x = np.linspace(0, 1, length)
        path = np.stack([x, np.full(length, offset), np.zeros(length)], 1)
        return {
            'state': [np.zeros(2, np.float32)],
            'action': [tag],
            'obs_extra_tcp_pose': list(path),
        }

    # Offsets from the expert path (0): 0.005 too close to the expert,
    # 0.05 kept, 0.06 too close to 0.05, 0.12 kept; slot 4 failed.
    batch = [
        (episode(0.0, 0), 0.0, True),
        (episode(0.005, 1, length=7), 0.0, True),
        (episode(0.05, 2), 0.0, True),
        (episode(0.06, 3), 0.0, True),
        (episode(0.3, 4), 0.0, False),
        (episode(0.12, 5), 0.0, True),
    ]
    groups, similar = _collect.select_alternatives(
        batch, 6, 3, min_distance=0.02
    )
    assert [r[2]['action'][0] for r in groups[0]] == [0, 2, 5]
    assert [r[1] for r in groups[0]] == [-1, 0, 1]
    assert similar == 2
    assert _collect.path_distance(
        np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0.5, 1], [1, 1]])
    ) == pytest.approx(1.0)
