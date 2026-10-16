#!/usr/bin/env python3
"""Collect prefixed evaluation directories into a Markdown comparison report.

Example (run from the repository root)::

    python scripts/visualization/gather_eval_results.py /path/to/checkpoints \
        --prefix cube_ --output results.md

Searches recursively, uses the latest record in each *_results.txt file, and
copies up to two success and two failure videos per evaluation beside the
report. Shared settings go into a linked *_shared_parameters.md file; the
main report compares differing model, planning, and evaluation parameters.
Add --landscape-dir and --expert-dir to include saved diagnostic experiments.
Reports use run names and short filenames instead of absolute local paths.
Episode errors are compared by expert/eval seed, with mean errors per run.
Videos use HTML players. Requires PyYAML and matplotlib for error plots;
no model, GPU, or video decoder is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote

import yaml

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
OUTCOME = re.compile(r'(?:success_|suc_)(true|false)\.mp4$', re.IGNORECASE)
INDEX = re.compile(r'^(?:episode|env)_(\d+)\.mp4$', re.IGNORECASE)
RECORD = re.compile(
    r'^==== CONFIG ====\s*\n(.*?)^==== RESULTS ====\s*\n(.*?)(?='
    r'^==== CONFIG ====|\Z)',
    re.MULTILINE | re.DOTALL,
)
ABSOLUTE_PATH = re.compile(
    r"""(?<![\w:/\\<])(?:"""
    r"""(?P<quote>['"])(?P<quoted>(?:/|~[/\\]|[A-Za-z]:[/\\])"""
    r"""[^'"\n]*)(?P=quote)|"""
    r"""(?P<bare>(?:/|~[/\\]|[A-Za-z]:[/\\])"""
    r"""[^\s'"`<>\[\]{},;|()]+))"""
)
ERROR_METRICS = {
    'xy_error_m': 'T XY error (m)',
    'eef_position_error_m': 'EEF position error (m)',
    'yaw_error_rad': 'T yaw error (rad)',
}


@dataclass
class Evaluation:
    source: Path
    config: dict[str, Any]
    config_text: str
    results_text: str
    success_rate: float
    episode_successes: list[bool]
    record_count: int
    metric_arrays: dict[str, list[int | float]] = field(default_factory=dict)
    mean_errors: dict[str, float] = field(default_factory=dict)


def numeric_array(text: str, key: str) -> list[int | float]:
    """Read complete numeric lists/NumPy arrays, preserving seed integers."""
    match = re.search(
        rf"['\"]{re.escape(key)}['\"]\s*:\s*"
        r'(?:array\(\s*)?\[([^\]]*)\]',
        text,
        re.DOTALL,
    )
    if not match:
        return []
    tokens = re.split(r'[,\s]+', match.group(1).strip().strip(','))
    if not tokens or not all(
        re.fullmatch(NUMBER, token)
        or token.lower() in {'nan', 'inf', '+inf', '-inf', 'none'}
        for token in tokens
    ):
        return []
    return [
        int(token)
        if re.fullmatch(r'[-+]?\d+', token)
        else float('nan')
        if token.lower() == 'none'
        else float(token)
        for token in tokens
    ]


def parse_result(path: Path) -> Evaluation:
    """Read the latest appended record without executing metric expressions."""
    records = list(RECORD.finditer(path.read_text(encoding='utf-8')))
    if not records:
        raise ValueError('missing CONFIG / RESULTS sections')
    config_text, results_text = records[-1].groups()
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise TypeError('CONFIG must be a YAML mapping')
    match = re.search(
        rf"['\"]success_rate['\"]\s*:\s*"
        rf'(?:np\.float(?:32|64)\()?({NUMBER})',
        results_text,
    )
    if not match:
        raise ValueError('missing numeric success_rate')
    rate = float(match.group(1))
    if not math.isfinite(rate) or not 0 <= rate <= 100:
        raise ValueError('success_rate must be between 0 and 100 percent')

    # NumPy prints arrays as Python-like text, not JSON. Parse only complete
    # boolean / 0-or-1 arrays; ellipsized arrays cannot map video indices safely.
    match = re.search(
        r"['\"]episode_successes['\"]\s*:\s*"
        r'(?:array\(\s*)?\[([^\]]*)\]',
        results_text,
        re.DOTALL,
    )
    outcomes: list[bool] = []
    if match:
        tokens = re.split(r'[,\s]+', match.group(1).strip())
        if tokens and all(
            token.lower() in {'true', 'false'}
            or re.fullmatch(NUMBER, token)
            and float(token) in {0, 1}
            for token in tokens
        ):
            outcomes = [
                token.lower() == 'true'
                or (token.lower() != 'false' and float(token) == 1)
                for token in tokens
            ]
    arrays = {
        key: numeric_array(results_text, key)
        for key in (
            'seeds',
            'expert_seeds',
            'start_steps',
            'goal_steps',
            *(f'episode_{key}' for key in ERROR_METRICS),
        )
        if re.search(rf"['\"]{key}['\"]\s*:", results_text)
    }
    means = {}
    for key in ERROR_METRICS:
        match = re.search(
            rf"['\"]mean_{key}['\"]\s*:\s*"
            rf'(?:np\.float(?:32|64)\()?({NUMBER}|nan|[-+]?inf)'
            r'\)?(?=\s*[,}])',
            results_text,
            re.IGNORECASE,
        )
        if match:
            means[key] = float(match.group(1))
    return Evaluation(
        path,
        config,
        config_text.strip(),
        results_text.strip(),
        rate,
        outcomes,
        len(records),
        arrays,
        means,
    )


def discover(root: Path, prefix: str, result_pattern: str) -> list[Path]:
    """Match literal directory-name prefixes, including the root itself."""
    directories = [root] if root.name.startswith(prefix) else []
    directories.extend(
        path
        for path in root.rglob('*')
        if path.is_dir() and path.name.startswith(prefix)
    )
    return sorted(
        {
            path
            for directory in directories
            for path in directory.glob(result_pattern)
            if path.is_file()
        }
    )


def select_videos(evaluation: Evaluation) -> dict[bool, list[Path]]:
    """Select two videos per outcome, in filename order, from the same run."""
    selected: dict[bool, list[Path]] = {True: [], False: []}
    for path in sorted(evaluation.source.parent.glob('*.mp4')):
        match = OUTCOME.search(path.name)
        if match:
            success = match.group(1).lower() == 'true'
        else:
            match = INDEX.match(path.name)
            if not match or int(match.group(1)) >= len(
                evaluation.episode_successes
            ):
                continue
            success = evaluation.episode_successes[int(match.group(1))]
        if len(selected[success]) < 2:
            selected[success].append(path)
    return selected


def short_path(value: str) -> str:
    """Keep the parent name and filename, omitting machine-specific roots."""
    path = (
        PureWindowsPath(value)
        if re.match(r'[A-Za-z]:[/\\]', value)
        else Path(value)
    )
    parts = [part for part in path.parts if part not in {path.anchor, '~'}]
    return '/'.join(parts[-2:])


def portable_text(value: str) -> str:
    """Shorten absolute paths embedded in parameter values or diagnostics."""

    def shorten(match: re.Match[str]) -> str:
        raw = match['quoted'] or match['bare']
        delimiter = match['quote'] or ''
        return f'{delimiter}{short_path(raw)}{delimiter}'

    return ABSOLUTE_PATH.sub(shorten, value)


def cell(value: Any) -> str:
    text = str(value)
    if isinstance(value, (str, Path)) and (
        text.startswith(('/', '~/')) or re.match(r'[A-Za-z]:[/\\]', text)
    ):
        text = short_path(text)
    return (
        portable_text(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('|', '&#124;')
        .replace('\n', '<br>')
    )


def flatten_config(config: dict[str, Any], prefix: str = '') -> dict[str, Any]:
    """Flatten nested settings, retaining lists and empty mappings as values."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict) and value:
            flat.update(flatten_config(value, name))
        else:
            flat[name] = value
    return flat


def important_parameter(key: str) -> bool:
    """Include experiment settings, omitting output and infrastructure paths."""
    return key in {
        'policy',
        'seed',
        'bf16',
        'compile',
        'state_columns',
        'action_block',
        'horizon',
        'history_len',
        'controller_scale',
        'grid_size',
        'samples',
        'normalize_reps',
        'goal_source',
        'goal_action_index',
        'dataset_name',
        'episode',
        'start_step',
        'goal_offset_steps',
        'play_in_reverse',
    } or (
        key.startswith(
            (
                'world.',
                'solver.',
                'objective.',
                'plan_config.',
                'eval.',
                'dataset.',
                'model.',
                'wm.',
            )
        )
        and key not in {'world.expert_output_dir'}
    )


def relative_url(path: Path, output: Path) -> str:
    relative = os.path.relpath(path, output.parent)
    return quote(Path(relative).as_posix(), safe='/.')


def experiment_files(directories: list[Path], name: str) -> list[Path]:
    """Accept individual experiment directories or trees of experiments."""
    return sorted(
        {
            path
            for directory in directories
            for path in directory.expanduser().resolve().rglob(name)
            if path.is_file()
        }
    )


def experiment_section(
    title: str,
    files: list[Path],
    output: Path,
    copy_assets: bool,
    warnings: list[str],
    landscape: bool = False,
) -> tuple[list[str], list[str]]:
    """Summarize saved diagnostic JSON and link or copy visualization assets."""
    rows = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or not isinstance(
                data.get('config', {}), dict
            ):
                raise TypeError('experiment metadata must contain mappings')
            required_key = (
                'best_grid_energy' if landscape else 'expert_selection_rate'
            )
            if required_key not in data:
                raise ValueError(
                    f'missing {required_key} in experiment results'
                )
            rows.append((path, data))
        except (OSError, TypeError, ValueError) as exc:
            warning = f'{path}: {exc}'
            warnings.append(warning)
            print(f'warning: skipping {warning}', file=sys.stderr)
    if not rows:
        return [], []
    configs = [flatten_config(data.get('config', {})) for _, data in rows]
    keys = sorted(set().union(*(config.keys() for config in configs)))
    shared = {
        key: configs[0][key]
        for key in keys
        if key != 'output.filename'
        and all(key in config for config in configs)
        and all(config[key] == configs[0][key] for config in configs[1:])
    }
    differing = [
        key for key in keys if key not in shared and important_parameter(key)
    ]
    common = ['', f'## {title}', '', '| Parameter | Value |', '| --- | --- |']
    common.extend(f'| {cell(k)} | {cell(v)} |' for k, v in shared.items())
    lines = ['', f'## {title}', '']
    if not landscape:
        lines.extend(
            [
                (
                    'Expert selection rate (0–1) measures how often the injected '
                    'expert sequence was selected. It does not by itself measure '
                    'task success.'
                ),
                '',
            ]
        )
    metric_keys = (
        [
            'best_grid_energy',
            'cem_energy',
            'ground_truth_action_energy',
            'grid_actions_with_lower_energy',
        ]
        if landscape
        else ['expert_selection_rate', 'num_replans', 'mode', 'source']
    )
    headers = ['Experiment', *metric_keys, *differing]
    lines.extend(
        [
            '| ' + ' | '.join(map(cell, headers)) + ' |',
            '| ' + ' | '.join('---' for _ in headers) + ' |',
        ]
    )
    for (path, data), config in zip(rows, configs):
        values = [cell(path.parent.name)]
        values.extend(cell(data.get(key, '—')) for key in metric_keys)
        values.extend(cell(config.get(key, '—')) for key in differing)
        lines.append('| ' + ' | '.join(values) + ' |')
    for index, (path, diagnostics) in enumerate(rows, 1):
        lines.extend(
            [
                '',
                f'### {cell(path.parent.name)}',
                '',
            ]
        )
        if landscape:
            video = diagnostics.get('comparison_video')
            if isinstance(video, str):
                source = (path.parent / video).resolve()
                if (
                    source.is_relative_to(path.parent.resolve())
                    and source.is_file()
                ):
                    target = source
                    if copy_assets:
                        target = (
                            output.parent
                            / f'{output.stem}_videos'
                            / f'landscape_{index:03d}'
                            / source.name
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if source != target.resolve():
                            shutil.copy2(source, target)
                    lines.extend(
                        [
                            '**Minimum-energy grid action | Goal**',
                            '',
                            '<video controls width="640" preload="none">',
                            f'  <source src="{relative_url(target, output)}" type="video/mp4">',
                            '</video>',
                            '',
                        ]
                    )
                else:
                    warnings.append(
                        f'{path}: comparison video missing or outside experiment: {source}'
                    )
            images = [
                path.parent / name
                for name in (
                    'landscape.png',
                    'landscape_3d.png',
                    'trajectory.png',
                    'start.png',
                    'goal.png',
                )
                if (path.parent / name).is_file()
            ]
            if not images:
                lines.extend(['No saved PNG visualizations found.', ''])
            for source in images:
                target = source
                if copy_assets:
                    target = (
                        output.parent
                        / f'{output.stem}_figures'
                        / f'landscape_{index:03d}'
                        / source.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.resolve() != target.resolve():
                        shutil.copy2(source, target)
                lines.extend(
                    [
                        f'![{cell(source.stem)}]({relative_url(target, output)})',
                        '',
                    ]
                )
        else:
            comparisons = diagnostics.get('comparison_videos', [])
            for video_index, comparison in enumerate(comparisons, 1):
                if not isinstance(comparison, dict) or not isinstance(
                    comparison.get('video'), str
                ):
                    continue
                source = (path.parent / comparison['video']).resolve()
                if (
                    not source.is_relative_to(path.parent.resolve())
                    or not source.is_file()
                ):
                    warnings.append(
                        f'{path}: comparison video missing or outside experiment: {source}'
                    )
                    continue
                target = source
                if copy_assets:
                    target = (
                        output.parent
                        / f'{output.stem}_videos'
                        / f'expert_{index:03d}'
                        / f'comparison_{video_index:03d}'
                        / source.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source != target.resolve():
                        shutil.copy2(source, target)
                label = (
                    f'Expert / selected / goal — environment '
                    f'{comparison.get("env_index", "—")}'
                )
                lines.extend(
                    [
                        f'**{cell(label)}**',
                        '',
                        '<video controls width="960" preload="none">',
                        f'  <source src="{relative_url(target, output)}" type="video/mp4">',
                        '</video>',
                        '',
                    ]
                )
                outcomes = comparison.get('outcomes', {})
                if isinstance(outcomes, dict):
                    lines.extend(
                        ['| Sequence | Final goal success |', '| --- | --- |']
                    )
                    for name, outcome in outcomes.items():
                        if isinstance(outcome, dict):
                            lines.append(
                                f'| {cell(name)} | {cell(outcome.get("final_goal_success", "—"))} |'
                            )
                    lines.append('')
            selections = diagnostics.get('selections', [])
            if isinstance(selections, list) and selections:
                first_iterations = {}
                last_iterations = {}
                for row in diagnostics.get('iterations', []):
                    if not isinstance(row, dict) or not isinstance(
                        row.get('iteration'), int
                    ):
                        continue
                    key = (row.get('replan_index'), row.get('env_index'))
                    if row['iteration'] == 0:
                        first_iterations[key] = row
                    if row['iteration'] > last_iterations.get(key, {}).get(
                        'iteration', -1
                    ):
                        last_iterations[key] = row
                fields = [
                    'env_index',
                    'env_step',
                    'expert_selected',
                    'initial_expert_rank',
                    'last_iteration_expert_rank',
                    'expert_tied_for_best',
                    'native_cem_mean_expert_rmse',
                ]
                if len(selections) > 20:
                    lines.extend(
                        [
                            (
                                f'Showing the first 20 of {len(selections)} '
                                'selections.'
                            ),
                            '',
                        ]
                    )
                lines.extend(
                    [
                        (
                            'Expert ranks use the sampled candidates at '
                            'iterations 0 and the last optimization iteration; '
                            'rank 1 is best (ties share rank).'
                        ),
                        '',
                        '| ' + ' | '.join(fields) + ' |',
                        '| ' + ' | '.join('---' for _ in fields) + ' |',
                    ]
                )
                for selection in selections[:20]:
                    if isinstance(selection, dict):
                        selection = dict(selection)
                        key = (
                            selection.get('replan_index'),
                            selection.get('env_index'),
                        )
                        selection.setdefault(
                            'initial_expert_rank',
                            first_iterations.get(key, {}).get(
                                'expert_rank', '—'
                            ),
                        )
                        selection.setdefault(
                            'last_iteration_expert_rank',
                            last_iterations.get(key, {}).get(
                                'expert_rank', '—'
                            ),
                        )
                        lines.append(
                            '| '
                            + ' | '.join(
                                cell(selection.get(key, '—')) for key in fields
                            )
                            + ' |'
                        )
    return lines, common


def error_plots(evaluations: list[Evaluation], output: Path) -> list[str]:
    """Compare final episode errors on matching seeds and saved run means."""
    metrics = [
        key
        for key in ERROR_METRICS
        if any(
            f'episode_{key}' in evaluation.metric_arrays
            or key in evaluation.mean_errors
            for evaluation in evaluations
        )
    ]
    if not metrics:
        return []
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lines = [
        '',
        '## Error comparison',
        '',
        (
            'Episode plots compare final goal errors; lower is better. '
            'The x-axis shows the evaluation seed, followed by the expert '
            'seed only when it differs. Episodes are matched using both '
            'seeds and saved start/goal steps. '
            'Mean plots use the saved mean across each run’s episodes. '
            'Missing/nonfinite values are omitted; runs without complete '
            'seed arrays are omitted from episode plots.'
        ),
        '',
    ]
    figure_dir = output.parent / f'{output.stem}_figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    for key in metrics:
        series = []
        for evaluation in evaluations:
            arrays = evaluation.metric_arrays
            errors = arrays.get(f'episode_{key}', [])
            seeds = arrays.get('seeds', [])
            metadata = [
                arrays.get(name, [None] * len(errors))
                for name in ('expert_seeds', 'start_steps', 'goal_steps')
            ]
            if (
                not errors
                or any(
                    len(values) != len(errors) for values in [seeds, *metadata]
                )
                or any(
                    value is not None
                    and (not math.isfinite(value) or int(value) != value)
                    for values in [seeds, *metadata]
                    for value in values
                )
            ):
                series.append([])
                continue
            series.append(
                [
                    (
                        tuple(int(v) if v is not None else None for v in case),
                        error,
                    )
                    for case, error in zip(zip(seeds, *metadata), errors)
                ]
            )
        cases = sorted(
            {case for points in series for case, _ in points},
            key=lambda case: tuple(
                -math.inf if value is None else value for value in case
            ),
        )
        positions = {case: index for index, case in enumerate(cases)}
        labels = [
            str(seed)
            if expert is None or expert == seed
            else f'{seed} / {expert}'
            for seed, expert, _, _ in cases
        ]
        fig, (episodes_ax, means_ax) = plt.subplots(
            2,
            1,
            figsize=(max(9, len(cases) * 0.5), 9),
            constrained_layout=True,
        )
        for index, (evaluation, points) in enumerate(zip(evaluations, series)):
            color = f'C{index % 10}'
            finite = [
                (case, value) for case, value in points if math.isfinite(value)
            ]
            episodes_ax.scatter(
                [positions[case] for case, _ in finite],
                [value for _, value in finite],
                label=f'Run {index + 1}',
                color=color,
                marker=('o', 's', '^', 'D', 'v', 'P', 'X')[index % 7],
                alpha=0.8,
            )
            mean = evaluation.mean_errors.get(key, math.nan)
            if math.isfinite(mean):
                means_ax.bar(index, mean, color=color)
        episodes_ax.set_xticks(
            range(len(cases)), labels, rotation=45, ha='right', fontsize=8
        )
        episodes_ax.set_xlabel('Eval seed / expert seed (if different)')
        episodes_ax.set_title('Per-episode final error')
        if cases:
            episodes_ax.legend(fontsize=8)
        else:
            episodes_ax.text(
                0.5,
                0.5,
                'No complete episode errors and seed arrays',
                ha='center',
                transform=episodes_ax.transAxes,
            )
        means_ax.set_xticks(
            range(len(evaluations)),
            [f'Run {index}' for index in range(1, len(evaluations) + 1)],
        )
        means_ax.set_title('Mean final error across episodes')
        for ax in (episodes_ax, means_ax):
            ax.set_ylabel(ERROR_METRICS[key])
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=0.25)
        target = figure_dir / f'{key}.png'
        try:
            fig.savefig(target, dpi=150)
        finally:
            plt.close(fig)
        lines.extend(
            [f'![{ERROR_METRICS[key]}]({relative_url(target, output)})', '']
        )
    return lines


def write_report(
    evaluations: list[Evaluation],
    output: Path,
    root: Path,
    prefix: str,
    copy_videos: bool = True,
    warnings: list[str] | None = None,
    landscape_dirs: list[Path] | None = None,
    expert_dirs: list[Path] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    warnings = list(warnings or [])
    configs = [flatten_config(evaluation.config) for evaluation in evaluations]
    keys = sorted(set().union(*(config.keys() for config in configs)))
    shared = {
        key: configs[0][key]
        for key in keys
        if key != 'output.filename'
        and all(key in config for config in configs)
        and all(config[key] == configs[0][key] for config in configs[1:])
    }
    differing = [
        key for key in keys if key not in shared and important_parameter(key)
    ]
    shared_path = output.with_name(f'{output.stem}_shared_parameters.md')
    shared_lines = [
        '# Shared evaluation parameters',
        '',
        (
            f'Settings present with the same value in all {len(evaluations)} '
            'evaluations.'
        ),
        '',
        f'[Back to evaluation results]({relative_url(output, shared_path)})',
        '',
    ]
    if shared:
        shared_lines.extend(['| Parameter | Value |', '| --- | --- |'])
        shared_lines.extend(
            f'| {cell(key)} | {cell(value)} |' for key, value in shared.items()
        )
    else:
        shared_lines.append('No parameters are shared across all evaluations.')
    mean_keys = [
        key
        for key in ERROR_METRICS
        if any(key in evaluation.mean_errors for evaluation in evaluations)
    ]
    headers = [
        'Evaluation',
        'Success rate (%)',
        *(f'Mean {ERROR_METRICS[key]}' for key in mean_keys),
        *map(cell, differing),
    ]
    lines = [
        '# Evaluation results',
        '',
        f'Evaluation directory names start with `{prefix}`.',
        '',
        (
            'Success rates are percentages. Each run shows its latest '
            'evaluation. Videos are selected in filename order '
            '(up to two per outcome).'
        ),
        '',
        f'[Shared parameters]({relative_url(shared_path, output)})',
        '',
        (
            'Only differing model, planning, and evaluation settings are shown '
            'below. A dash means the parameter was absent from that run.'
        ),
        '',
        '| ' + ' | '.join(headers) + ' |',
        '| '
        + ' | '.join(
            [
                '---',
                '---:',
                *['---:'] * len(mean_keys),
                *['---'] * len(differing),
            ]
        )
        + ' |',
    ]
    for index, (evaluation, config) in enumerate(zip(evaluations, configs), 1):
        name = evaluation.source.parent.relative_to(root).as_posix()
        values = [
            f'Run {index} — {cell(name)}',
            f'{evaluation.success_rate:.2f}',
            *(
                f'{evaluation.mean_errors[key]:.6g}'
                if key in evaluation.mean_errors
                and math.isfinite(evaluation.mean_errors[key])
                else '—'
                for key in mean_keys
            ),
            *(cell(config.get(key, '—')) for key in differing),
        ]
        lines.append('| ' + ' | '.join(values) + ' |')

    lines.extend(error_plots(evaluations, output))

    for index, evaluation in enumerate(evaluations, 1):
        lines.extend(
            [
                '',
                f'## Run {index}',
                '',
            ]
        )
        if evaluation.record_count > 1:
            lines.extend(
                [
                    (
                        f'Using record {evaluation.record_count} of '
                        f'{evaluation.record_count}. Videos reflect the files '
                        'currently in the evaluation directory.'
                    ),
                    '',
                ]
            )
        videos = select_videos(evaluation)
        for success, label in ((False, 'Failure'), (True, 'Success')):
            lines.extend([f'### {label} videos', ''])
            for video_index, source in enumerate(videos[success], 1):
                target = source
                if copy_videos:
                    target = (
                        output.parent
                        / f'{output.stem}_videos'
                        / f'run_{index:03d}'
                        / source.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.resolve() != target.resolve():
                        shutil.copy2(source, target)
                lines.extend(
                    [
                        f'**{label} {video_index}**',
                        '',
                        '<video controls width="640" preload="none">',
                        (
                            f'  <source src="{relative_url(target, output)}" '
                            'type="video/mp4">'
                        ),
                        '  Your viewer does not support embedded video.',
                        '</video>',
                        '',
                    ]
                )
            if len(videos[success]) < 2:
                lines.append(
                    f'Only {len(videos[success])} of 2 requested '
                    f'{label.lower()} videos available. Unclassified videos '
                    'are omitted.'
                )
            lines.append('')
        lines.extend(
            [
                '### Metrics',
                '',
                '```text',
                portable_text(evaluation.results_text),
                '```',
            ]
        )
    expert_files = sorted(
        {
            *experiment_files(expert_dirs or [], 'expert_candidates.json'),
            *(
                evaluation.source.parent / 'expert_candidates.json'
                for evaluation in evaluations
                if (
                    evaluation.source.parent / 'expert_candidates.json'
                ).is_file()
            ),
        }
    )
    for title, files, landscape in (
        (
            'Action-space energy landscapes',
            experiment_files(landscape_dirs or [], 'results.json'),
            True,
        ),
        ('Expert-action selection', expert_files, False),
    ):
        section, common = experiment_section(
            title, files, output, copy_videos, warnings, landscape
        )
        lines.extend(section)
        shared_lines.extend(common)
    shared_path.write_text('\n'.join(shared_lines) + '\n', encoding='utf-8')
    if warnings:
        lines.extend(['', '## Skipped result files', ''])
        lines.extend(
            f'- {cell(portable_text(warning))}' for warning in warnings
        )
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'root', type=Path, help='Directory searched recursively'
    )
    parser.add_argument(
        '--prefix', required=True, help='Literal evaluation directory prefix'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('results.md'),
        help='Markdown output (default: results.md)',
    )
    parser.add_argument(
        '--result-pattern',
        default='*_results.txt',
        help='Result filename glob within each matched directory',
    )
    parser.add_argument(
        '--link-videos',
        action='store_true',
        help='Link original videos and figures instead of copying them',
    )
    parser.add_argument(
        '--landscape-dir',
        type=Path,
        action='append',
        default=[],
        help='Energy-landscape output directory or tree (repeatable)',
    )
    parser.add_argument(
        '--expert-dir',
        type=Path,
        action='append',
        default=[],
        help='Expert-candidate output directory or tree (repeatable)',
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    for directory in [*args.landscape_dir, *args.expert_dir]:
        if not directory.expanduser().is_dir():
            parser.error(f'experiment directory does not exist: {directory}')
    if not root.is_dir():
        parser.error(f'root is not a directory: {root}')
    if not args.prefix:
        parser.error('--prefix must not be empty')
    evaluations: list[Evaluation] = []
    warnings: list[str] = []
    for path in discover(root, args.prefix, args.result_pattern):
        try:
            evaluations.append(parse_result(path))
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            warning = f'{path}: {exc}'
            warnings.append(warning)
            print(f'warning: skipping {warning}', file=sys.stderr)
    if not evaluations:
        raise SystemExit('No valid results in matching evaluation directories')
    write_report(
        evaluations,
        args.output.expanduser().resolve(),
        root,
        args.prefix,
        copy_videos=not args.link_videos,
        warnings=warnings,
        landscape_dirs=args.landscape_dir,
        expert_dirs=args.expert_dir,
    )
    print(f'Wrote {len(evaluations)} evaluations to {args.output}')


if __name__ == '__main__':
    main()
