"""Evaluation reports preserve run settings and select classified videos."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.visualization.gather_eval_results import (
    discover,
    episode_results,
    parse_result,
    write_report,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / 'scripts/visualization/gather_eval_results.py'
)


def result(directory, rate=50, outcomes='True, False, True, False'):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'cube_results.txt'
    path.write_text(
        '==== CONFIG ====\n'
        'policy: models/checkpoint.pt\n'
        'plan_config:\n  horizon: 5\n  receding_horizon: 2\n'
        'solver:\n  num_samples: 128\n'
        '==== RESULTS ====\n'
        f"metrics: {{'success_rate': {rate}, "
        f"'episode_successes': array([{outcomes}])}}\n"
        'evaluation_time: 1.5 seconds\n',
        encoding='utf-8',
    )
    return path


def test_discovery_uses_literal_prefix_and_includes_root(tmp_path):
    first = result(tmp_path / 'cube_[one]')
    second = result(tmp_path / 'model' / 'evals' / 'cube_[two]')
    result(tmp_path / 'other_cube_[three]')
    assert discover(tmp_path, 'cube_[', '*_results.txt') == [first, second]
    assert discover(first.parent, 'cube_[', '*_results.txt') == [first]


def test_latest_record_and_complete_numpy_arrays(tmp_path):
    path = result(tmp_path)
    previous = path.read_text()
    result(tmp_path, rate='np.float64(75.0)', outcomes='1., 0.,\n 1., 1.')
    path.write_text(previous + path.read_text())
    evaluation = parse_result(path)
    assert evaluation.record_count == 2
    assert evaluation.success_rate == 75
    assert evaluation.episode_successes == [True, False, True, True]
    assert evaluation.config['plan_config']['receding_horizon'] == 2


def error_result(directory, metrics):
    path = result(directory)
    path.write_text(
        path.read_text().replace(
            'evaluation_time:', f'error_metrics: {metrics}\nevaluation_time:'
        )
    )
    return path


def test_error_metrics_parse_latest_record_without_executing_text(tmp_path):
    path = error_result(tmp_path, "{'mean_xy_error_m': 99}")
    previous = path.read_text()
    error_result(
        tmp_path,
        """{
        'seeds': array([2, 9007199254740993], dtype=int64),
        'expert_seeds': [200, 300], 'start_steps': [0, 10],
        'goal_steps': [10, 20],
        'episode_xy_error_m': array([1.e-2, 2.e-2], dtype=float32),
        'episode_eef_position_error_m': [nan, 0.03],
        'episode_yaw_error_rad': [0.1, inf],
        'mean_xy_error_m': np.float64(1.5e-2),
        'mean_eef_position_error_m': nan,
        'mean_yaw_error_rad': __import__('os').system('false')
    }""",
    )
    path.write_text(previous + path.read_text())
    evaluation = parse_result(path)
    assert evaluation.metric_arrays['seeds'] == [2, 9007199254740993]
    assert evaluation.metric_arrays['expert_seeds'] == [200, 300]
    assert evaluation.metric_arrays['episode_xy_error_m'] == [0.01, 0.02]
    assert evaluation.mean_errors['xy_error_m'] == 0.015
    assert str(evaluation.mean_errors['eef_position_error_m']) == 'nan'
    assert 'yaw_error_rad' not in evaluation.mean_errors


def test_report_plots_align_seed_pairs_and_steps(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from PIL import Image

    first = parse_result(
        error_result(
            tmp_path / 'cube_a',
            """{
        'seeds': [10, 2, 4, 4], 'expert_seeds': [100, 2, 40, 40],
        'start_steps': [0, 0, 0, 1], 'goal_steps': [5, 5, 5, 6],
        'episode_xy_error_m': [0.1, 0.2, 0.3, 0.4],
        'episode_eef_position_error_m': [nan, 0.1, 0.2, 0.3],
        'episode_yaw_error_rad': [0.1, 0.2, 0.3, 0.4],
        'mean_xy_error_m': 0.25, 'mean_eef_position_error_m': nan,
        'mean_yaw_error_rad': 0.25
    }""",
        )
    )
    second = parse_result(
        error_result(
            tmp_path / 'cube_b',
            """{
        'seeds': [2, 10], 'expert_seeds': [2, 101],
        'start_steps': [0, 0], 'goal_steps': [5, 5],
        'episode_xy_error_m': [0.15, 0.05], 'mean_xy_error_m': 0.1
    }""",
        )
    )
    points = []
    means = []
    episode_axes = []
    scatter, bar = Axes.scatter, Axes.bar

    def capture_scatter(ax, x, y, **kwargs):
        points.append((x, y, kwargs['label']))
        episode_axes.append(ax)
        return scatter(ax, x, y, **kwargs)

    def capture_bar(ax, x, height, **kwargs):
        means.append((x, height))
        return bar(ax, x, height, **kwargs)

    monkeypatch.setattr(Axes, 'scatter', capture_scatter)
    monkeypatch.setattr(Axes, 'bar', capture_bar)
    output = tmp_path / 'report' / 'comparison.md'
    write_report([first, second], output, tmp_path, 'cube_', copy_videos=False)
    report = output.read_text()
    assert '## Error comparison' in report
    assert 'Mean T XY error (m)' in report
    assert '| 0.25 | — | 0.25 |' in report
    # Only XY has competing runs; the lower mean is underlined.
    assert '| <u>0.1</u> | — | — |' in report
    assert points[0] == ([3, 0, 1, 2], [0.1, 0.2, 0.3, 0.4], 'Run 1')
    assert points[1] == ([0, 4], [0.15, 0.05], 'Run 2')
    assert points[2] == ([0, 1, 2], [0.1, 0.2, 0.3], 'Run 1')
    assert means == [(0, 0.25), (1, 0.1), (0, 0.25)]
    assert [tick.get_text() for tick in episode_axes[0].get_xticklabels()] == [
        '2',
        '4 / 40',
        '4 / 40',
        '10 / 100',
        '10 / 101',
    ]
    assert (
        episode_axes[0].get_xlabel()
        == 'Eval seed / expert seed (if different)'
    )
    figures = re.findall(r'!\[[^\]]+\]\(([^)]+)\)', report)
    assert len(figures) == 4  # three metrics plus the mean-error lines
    assert figures[-1].endswith('mean_errors.png')
    for figure in figures:
        with Image.open(output.parent / figure) as image:
            image.verify()


@pytest.mark.parametrize(
    'metadata',
    [
        "'seeds': [1, ..., 3]",
        "'seeds': [1]",
        "'seeds': [nan, 2]",
        "'seeds': [1.5, 2]",
        "'seeds': [1, 2], 'expert_seeds': [10, ..., 20]",
        '',
    ],
)
def test_incomplete_seed_metadata_keeps_means_without_guessing_cases(
    tmp_path, metadata, monkeypatch
):
    from matplotlib.axes import Axes

    evaluation = parse_result(
        error_result(
            tmp_path / 'cube_a',
            "{'episode_xy_error_m': [0.1, 0.2], 'mean_xy_error_m': 0.15, "
            + metadata
            + '}',
        )
    )
    points = []
    scatter = Axes.scatter

    def capture(ax, x, y, **kwargs):
        points.append((x, y))
        return scatter(ax, x, y, **kwargs)

    monkeypatch.setattr(Axes, 'scatter', capture)
    output = tmp_path / 'comparison.md'
    write_report([evaluation], output, tmp_path, 'cube_')
    assert points == [([], [])]
    assert '| 0.15 |' in output.read_text()
    assert (tmp_path / 'comparison_figures/xy_error_m.png').is_file()


def test_cli_error_plots_support_eval_seeds_without_expert_metadata(tmp_path):
    error_result(
        tmp_path / 'cube_a',
        """{
        'seeds': array([2, 3]), 'episode_xy_error_m': [0.1, 0.2],
        'mean_xy_error_m': 0.15
    }""",
    )
    output = tmp_path / 'comparison.md'
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            '--prefix',
            'cube_',
            '-o',
            str(output),
            '--link-videos',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'comparison_figures/xy_error_m.png' in output.read_text()


@pytest.mark.parametrize('outcomes', ['True, ..., False', '', '1, 2, 0'])
def test_incomplete_outcomes_do_not_guess_episodes(tmp_path, outcomes):
    evaluation = parse_result(result(tmp_path, outcomes=outcomes))
    (tmp_path / 'env_0.mp4').touch()
    (tmp_path / 'episode_remaining_0.mp4').touch()
    assert evaluation.episode_successes == []
    assert episode_results([evaluation]) == ([], [])


def test_episode_videos_support_both_naming_styles(tmp_path):
    evaluation = parse_result(result(tmp_path, outcomes='True, False'))
    for name in [
        'episode_0.mp4',
        'env_1_success_True.mp4',
        'env_99.mp4',
        'episode_remaining_0.mp4',
        'unknown.mp4',
    ]:
        (tmp_path / name).touch()
    cases, rows = episode_results([evaluation])
    assert cases == [(None, None, None, None, 0), (None, None, None, None, 1)]
    assert [row[0].video.name for row in rows] == [
        'episode_0.mp4',
        'env_1_success_True.mp4',
    ]
    # Recorded outcomes win over the filename.
    assert [row[0].success for row in rows] == [True, False]


def test_episodes_match_across_runs_by_seed_and_steps(tmp_path):
    first = parse_result(
        error_result(
            tmp_path / 'cube_a',
            """{
        'seeds': [1, 2, 3, 4], 'start_steps': [0, 0, 0, 0],
        'goal_steps': [5, 5, 5, 5],
        'episode_xy_error_m': [0.1, 0.2, 0.3, 0.4]
    }""",
        )
    )
    second = parse_result(
        error_result(
            tmp_path / 'cube_b',
            """{
        'seeds': [4, 3, 2, 9], 'start_steps': [0, 0, 0, 0],
        'goal_steps': [5, 5, 5, 5],
        'episode_xy_error_m': [0.04, 0.03, 0.02, 0.09]
    }""",
        )
    )
    (first.source.parent / 'episode_3.mp4').touch()
    (second.source.parent / 'episode_0.mp4').touch()
    cases, rows = episode_results([first, second])
    assert [case[0] for case in cases] == [1, 2, 3, 4, 9]
    assert rows[0][1] is None and rows[4][0] is None
    seed_4 = rows[3]
    assert seed_4[0].errors == {'xy_error_m': 0.4}
    assert seed_4[1].errors == {'xy_error_m': 0.04}
    assert seed_4[0].video.parent.name == 'cube_a'
    assert seed_4[1].video.parent.name == 'cube_b'
    assert [r.success for r in seed_4] == [False, True]

    output = tmp_path / 'report' / 'results.md'
    write_report([first, second], output, tmp_path, 'cube_')
    report = output.read_text()
    assert '## Episode outcomes' in report
    assert '### Outcomes differ across runs (3)' in report
    assert (
        '| [seed 4 · steps 0→5](results_episodes.md#episode-3) | 1/2 | ✗ | ✓ |'
    ) in report
    assert (
        '| [seed 9 · steps 0→5](results_episodes.md#episode-4) | 0/1 |  | ✗ |'
    ) in report
    episodes = output.with_name('results_episodes.md').read_text()
    assert '<a id="episode-3"></a>' in episodes
    assert 'results_videos/run_001/episode_3.mp4' in episodes
    assert 'results_videos/run_002/episode_0.mp4' in episodes
    html = output.with_suffix('.html').read_text()
    data = json.loads(re.search(r'const DATA = (.*);\n', html)[1])
    assert [run['name'] for run in data['runs']] == ['cube_a', 'cube_b']
    assert data['episodes'][3]['case']['seed'] == 4
    assert data['episodes'][3]['group'] == 'differs'
    assert [r['video'] for r in data['episodes'][3]['results']] == [
        'results_videos/run_001/episode_3.mp4',
        'results_videos/run_002/episode_0.mp4',
    ]
    assert str(tmp_path) not in html


@pytest.mark.parametrize('copy_videos', [True, False])
def test_report_embeds_videos_and_keeps_config(tmp_path, copy_videos):
    root = tmp_path / 'inputs'
    evaluation = parse_result(result(root / 'cube_one'))
    (evaluation.source.parent / 'case 0_suc_True.mp4').write_bytes(b'x')
    for index in range(4):
        (evaluation.source.parent / f'episode_{index}.mp4').write_bytes(
            b'video'
        )
    output = tmp_path / 'report' / 'results.md'
    write_report([evaluation], output, root, 'cube_', copy_videos=copy_videos)
    report = output.read_text()
    assert '| 50.00 |' in report
    assert '[Shared parameters](results_shared_parameters.md)' in report
    assert evaluation.source.name not in report
    assert '| Run 1 — cube_one |' in report
    assert 'num_samples: 128' not in report
    assert 'evaluation_time' not in report
    shared = output.with_name('results_shared_parameters.md').read_text()
    assert '| solver.num_samples | 128 |' in shared
    assert '| policy | models/checkpoint.pt |' in shared
    assert '[Back to evaluation results](results.md)' in shared
    assert str(tmp_path) not in report + shared
    assert '| [episode 1](results_episodes.md#episode-1) | 0/1 | ✗ |' in report
    episodes = output.with_name('results_episodes.md').read_text()
    video_root = (
        'results_videos/run_001' if copy_videos else '../inputs/cube_one'
    )
    assert episodes.count('<video controls width="320"') == 4
    assert f'<source src="{video_root}/episode_1.mp4"' in episodes
    copied = list(output.parent.rglob('*.mp4'))
    assert len(copied) == (4 if copy_videos else 0)
    assert all(path.read_bytes() == b'video' for path in copied)
    assert output.with_suffix('.html').is_file()


def test_report_compares_differing_important_settings(tmp_path):
    first = parse_result(result(tmp_path / 'cube_a'))
    second = parse_result(result(tmp_path / 'cube_b', rate=75))
    second.config['policy'] = 'models/other.pt'
    second.config['plan_config']['horizon'] = 10
    second.config['solver']['num_samples'] = 256
    first.config['eval'] = {'source': 'env', 'start_from_beginning': False}
    second.config['eval'] = {'source': 'env'}
    first.config['subdir'] = 'cube_a'
    second.config['subdir'] = 'cube_b'
    first.config['world'] = {'options': {}, 'image_keys': ['pixels', 'goal']}
    second.config['world'] = {'options': {}, 'image_keys': ['pixels', 'goal']}
    first.config['output'] = {'filename': 'cube_results.txt'}
    second.config['output'] = {'filename': 'cube_results.txt'}
    output = tmp_path / 'comparison.md'
    write_report([first, second], output, tmp_path, 'cube_')
    report = output.read_text()
    shared = (tmp_path / 'comparison_shared_parameters.md').read_text()
    assert 'models/checkpoint.pt' in report and 'models/other.pt' in report
    assert 'plan_config.horizon' in report
    assert 'solver.num_samples' in report
    assert 'eval.start_from_beginning' in report
    assert 'False' in report and '—' in report
    assert 'subdir' not in report and 'subdir' not in shared
    assert 'plan_config.receding_horizon' not in report
    assert '| plan_config.receding_horizon | 2 |' in shared
    assert '| eval.source | env |' in shared
    assert '| world.options | {} |' in shared
    assert 'world.image_keys' in shared
    assert 'eval.start_from_beginning' not in shared
    assert 'solver.num_samples' not in shared
    assert 'output.filename' not in report + shared
    assert 'cube_results.txt' not in report + shared


def test_shared_report_handles_no_common_settings(tmp_path):
    first = parse_result(result(tmp_path / 'cube_a'))
    second = parse_result(result(tmp_path / 'cube_b'))
    first.config = {'policy': 'one.pt'}
    second.config = {'policy': 'two.pt'}
    output = tmp_path / 'comparison.md'
    write_report([first, second], output, tmp_path, 'cube_')
    shared = (tmp_path / 'comparison_shared_parameters.md').read_text()
    assert 'No parameters are shared' in shared
    assert 'one.pt' in output.read_text()


def test_cli_skips_bad_results_and_reports_missing_videos(tmp_path):
    result(tmp_path / 'cube_good')
    bad = result(tmp_path / 'cube_bad')
    bad.write_text('incomplete evaluation')
    output = tmp_path / 'results.md'
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            '--prefix',
            'cube_',
            '-o',
            str(output),
            '--link-videos',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'warning: skipping' in completed.stderr
    report = output.read_text()
    assert 'Skipped result files' in report
    assert 'cube_bad/cube_results.txt: missing CONFIG / RESULTS' in report
    assert str(tmp_path) not in report
    assert not (tmp_path / 'results_videos').exists()


def test_cli_no_matches_does_not_write_report(tmp_path):
    result(tmp_path / 'other_run')
    output = tmp_path / 'results.md'
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            '--prefix',
            'cube_',
            '-o',
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert 'No valid results' in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize('copy_assets', [True, False])
def test_report_includes_landscapes_and_expert_selection(
    tmp_path, copy_assets
):
    evaluation = parse_result(result(tmp_path / 'cube_eval'))
    diagnostics = {
        'config': evaluation.config,
        'expert_selection_rate': 0.5,
        'mode': 'mpc',
        'source': 'env',
        'num_replans': 2,
        'iterations': [
            {'env_index': 0, 'iteration': 0, 'expert_rank': 17},
            {'env_index': 0, 'iteration': 29, 'expert_rank': 3},
        ],
        'selections': [
            {
                'env_index': 0,
                'expert_selected': True,
                'expert_rank': 1,
                'expert_cost': 0.1,
                'selected_cost': 0.1,
            }
        ],
    }
    comparison = evaluation.source.parent / 'comparisons' / 'env_0.mp4'
    comparison.parent.mkdir()
    comparison.write_bytes(b'comparison video')
    diagnostics['comparison_videos'] = [
        {
            'video': 'comparisons/env_0.mp4',
            'env_index': 0,
            'replan_index': 1,
            'outcomes': {
                'expert': {'final_goal_success': True},
                'selected': {
                    'final_goal_success': False,
                    'final_goal_errors': {
                        'xy_error_m': 0.05,
                        'yaw_error_rad': 0.1,
                        'eef_position_error_m': 0.3,
                    },
                },
            },
        }
    ]
    expert_path = evaluation.source.parent / 'expert_candidates.json'
    expert_path.write_text(json.dumps(diagnostics))
    landscapes = tmp_path / 'landscapes'
    for index in range(2):
        directory = landscapes / f'landscape_{index}'
        directory.mkdir(parents=True)
        (directory / 'results.json').write_text(
            json.dumps(
                {
                    'config': {
                        'policy': 'model.pt',
                        'controller_scale': 0.1 + index,
                    },
                    'best_grid_energy': 0.2,
                    'comparison_video': 'best_action/env_0.mp4',
                    'cem_energy': 0.3,
                    'ground_truth_action_energy': 0.1,
                    'grid_actions_with_lower_energy': 0,
                }
            )
        )
        (directory / 'landscape.png').write_bytes(b'plot')
        (directory / 'best_action').mkdir()
        (directory / 'best_action' / 'env_0.mp4').write_bytes(b'video')
        (directory / 'landscape_3d.png').write_bytes(b'3d plot')
    output = tmp_path / 'report' / 'results.md'
    write_report(
        [evaluation],
        output,
        tmp_path,
        'cube_',
        copy_videos=copy_assets,
        landscape_dirs=[landscapes],
        expert_dirs=[evaluation.source.parent],
    )
    report = output.read_text()
    assert '## Action-space energy landscapes' in report
    assert '## Expert-action selection' in report
    assert 'controller_scale' in report and '1.1' in report
    assert 'expert_selection_rate' in report and '0.5' in report
    assert 'expert_rank' in report
    assert 'initial_expert_rank' in report
    assert 'last_iteration_expert_rank' in report
    assert '| 17 | 3 |' in report
    assert 'initial_expert_cost' not in report
    assert 'initial_cem_mean_cost' not in report
    assert '<video controls width="960"' in report
    assert 'Final goal success' in report
    assert '| expert | True |' in report
    assert '| selected | False |' in report
    # Only XY exceeds its tolerance; EEF has no tolerance unless configured.
    assert '| selected | False | 0.05 | 0.1 | 0.3 | T position |' in report
    comparison_copies = list((output.parent / 'results_videos').rglob('*.mp4'))
    assert len(comparison_copies) == (3 if copy_assets else 0)
    assert report.count('### cube_eval') == 1
    # Generic diagnostic dir names are labeled by their path under the root.
    assert '### landscapes/landscape_0' in report
    assert '![landscape]' in report and '![landscape_3d]' in report
    assert 'Minimum-energy grid action | Goal' in report
    assert (
        output.parent / 'results_videos/landscape_001/env_0.mp4'
    ).is_file() == copy_assets
    figures = list((output.parent / 'results_figures').rglob('*.png'))
    assert len(figures) == (4 if copy_assets else 0)
    shared = (output.parent / 'results_shared_parameters.md').read_text()
    assert '## Action-space energy landscapes' in shared
    assert '| policy | model.pt |' in shared
    assert 'controller_scale' not in shared
    assert '## Expert-action selection' in shared
    assert expert_path.name not in report
    links = re.findall(r'(?<!!)\[[^\]\n]+\]\(([^\n)]+)\)', report + shared)
    assert [link for link in links if '_episodes.md#' not in link] == [
        'results_shared_parameters.md',
        'results.md',
    ]
    assert str(tmp_path) not in report + shared
    html = output.with_suffix('.html').read_text()
    data = json.loads(re.search(r'const DATA = (.*);\n', html)[1])
    assert '<h2>Action-space energy landscapes</h2>' in data['diagnostics']
    assert '<h3>landscapes/landscape_0</h3>' in data['diagnostics']
    assert (
        'src="results_figures/landscape_001/landscape.png"'
        in data['diagnostics']
    ) == copy_assets
    assert '<td>policy</td><td>model.pt</td>' in data['diagnostic_shared']
    assert str(tmp_path) not in html


def test_cli_includes_external_expert_diagnostics_and_skips_bad_json(tmp_path):
    result(tmp_path / 'cube_eval')
    expert = tmp_path / 'external_experiment'
    expert.mkdir()
    (expert / 'expert_candidates.json').write_text(
        json.dumps(
            {
                'expert_selection_rate': 1.0,
                'config': {'policy': 'model.pt'},
                'selections': [
                    {'env_index': index, 'expert_selected': True}
                    for index in range(21)
                ],
            }
        )
    )
    landscape = tmp_path / 'bad_landscape'
    landscape.mkdir()
    (landscape / 'results.json').write_text('not JSON')
    output = tmp_path / 'results.md'
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            '--prefix',
            'cube_',
            '-o',
            str(output),
            '--expert-dir',
            str(expert),
            '--landscape-dir',
            str(landscape),
            '--link-videos',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = output.read_text()
    assert '## Expert-action selection' in report
    assert 'first 20 of 21 selections' in report
    assert 'see the full diagnostic results' not in report
    assert 'No saved PNG' not in report
    assert 'bad_landscape' in report
    assert 'warning: skipping' in completed.stderr
    assert str(tmp_path) not in report


@pytest.mark.parametrize(
    'path_root', ['/network/scratch/user', 'C:\\Users\\researcher', '~/models']
)
def test_report_shortens_paths_in_parameters_metrics_and_warnings(
    tmp_path, path_root
):
    first = parse_result(result(tmp_path / 'cube_a'))
    second = parse_result(result(tmp_path / 'cube_b'))
    first.config['policy'] = f'{path_root}/model_a/checkpoint.pt'
    second.config['policy'] = f'{path_root}/model_b/checkpoint.pt'
    for evaluation in (first, second):
        evaluation.config['world'] = {
            'expert_checkpoint': f'{path_root}/expert run/final.pt',
            'options': {'assets': [f'{path_root}/assets/object.xml']},
        }
        evaluation.results_text += (
            f"\ncheckpoint: '{evaluation.config['policy']}'"
        )
    output = tmp_path / 'comparison.md'
    write_report(
        [first, second],
        output,
        tmp_path,
        'cube_',
        warnings=[f"{path_root}/bad/results.txt: missing '{path_root}/data'"],
    )
    report = output.read_text()
    shared = output.with_name('comparison_shared_parameters.md').read_text()
    assert path_root not in report + shared
    assert str(tmp_path) not in report + shared
    assert 'model_a/checkpoint.pt' in report
    assert 'model_b/checkpoint.pt' in report
    assert 'expert run/final.pt' in shared
    assert 'assets/object.xml' in shared
    assert 'bad/results.txt: missing' in report
    assert first.config['policy'] == f'{path_root}/model_a/checkpoint.pt'
