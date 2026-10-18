#!/usr/bin/env python3
"""Collect prefixed evaluation directories into comparison reports.

Example (run from the repository root)::

    python scripts/visualization/gather_eval_results.py /path/to/checkpoints \
        --prefix cube_ --output results.md

Omit --prefix to gather every result file under the root.

Searches recursively, uses the latest record in each *_results.txt file and
writes two reports beside each other: results.html, an interactive report
(run selector, error charts, episode × run outcome matrix and a synced
episode video viewer; loads plotly.js from a CDN), and results.md with static
plots. Episodes are matched across runs by seeds and start/goal steps, and
results_episodes.md shows each episode's videos from every run side by side.
Episode videos are copied beside the reports (omit them with --no-videos). Shared settings go into a
linked *_shared_parameters.md file; the reports compare differing model,
planning, and evaluation parameters.
Add --landscape-dir and --expert-dir to include saved diagnostic experiments.
Reports use run names and short filenames instead of absolute local paths.
Episode errors are compared by expert/eval seed, with mean errors per run.
Videos use HTML players. Requires PyYAML and matplotlib for error plots;
no model, GPU, or video decoder is needed.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote, unquote

import yaml

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
# Episode videos: episode_<i>.mp4 or env_<i>[_success_<bool>].mp4.
VIDEO = re.compile(
    r'^(?:episode|env)_(\d+)(?:_(?:success|suc)_(true|false))?\.mp4$',
    re.IGNORECASE,
)
CASE_KEYS = ('seeds', 'expert_seeds', 'start_steps', 'goal_steps')
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
# Run directories named <model>_ep<N> become one line per model over epochs.
EPOCH = re.compile(r'^(.*)_ep(\d+)$')
# Checkpoint paths such as <model>/weights_epoch_<N>.pt yield model and epoch.
CHECKPOINT = re.compile(r'([^/\\]+)[/\\][^/\\]*?(epoch|step)_(\d+)\.\w+$')
FACET_NAMES = {
    'plan_config.horizon': 'horizon',
    'eval.goal_offset_steps': 'goal steps',
}
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


@dataclass
class EpisodeResult:
    success: bool | None
    errors: dict[str, float]
    video: Path | None


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


def discover(
    root: Path,
    prefix: str,
    result_pattern: str,
    exclude: list[str] | None = None,
) -> list[Path]:
    """Match literal directory-name prefixes, including the root itself.

    `exclude` globs are matched against directory paths relative to the root.
    """
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
            and not any(
                fnmatch.fnmatch(directory.relative_to(root).as_posix(), glob)
                for glob in exclude or []
            )
        }
    )


def run_facets(params: dict[str, str]) -> dict[str, str]:
    """Name differing settings for report filters; split checkpoint paths."""
    facets = {}
    for key, value in params.items():
        if match := CHECKPOINT.search(value):
            facets['model'] = match[1]
            facets[match[2]] = match[3]
        else:
            facets[FACET_NAMES.get(key, key.rsplit('.', 1)[-1])] = value
    return facets


def run_group(directory: Path, root: Path) -> tuple[str, int | None]:
    """Group <model>_ep<N> runs by their path with the epoch removed.

    Nested directories such as <model>_ep<N>/expert_candidates form their own
    group instead of merging with every other run's expert_candidates.
    """
    parts = list(directory.relative_to(root).parts) or [directory.name]
    for index in reversed(range(len(parts))):
        if match := EPOCH.match(parts[index]):
            parts[index] = match[1]
            return '/'.join(parts), int(match[2])
    return '/'.join(parts), None


def episode_cases(evaluation: Evaluation) -> list[tuple]:
    """Identify episodes by seeds and start/goal steps, else by index."""
    arrays = evaluation.metric_arrays
    count = len(evaluation.episode_successes) or max(
        (
            len(values)
            for key, values in arrays.items()
            if key.startswith('episode_')
        ),
        default=0,
    )
    columns = [arrays.get(key) for key in CASE_KEYS]
    if columns[0] and all(
        values is None
        or len(values) == count
        and all(math.isfinite(v) and int(v) == v for v in values)
        for values in columns
    ):
        return [
            (*(None if v is None else int(v[i]) for v in columns), None)
            for i in range(count)
        ]
    return [(None, None, None, None, i) for i in range(count)]


def case_label(case: tuple) -> str:
    seed, expert, start, goal, index = case
    if seed is None:
        return f'episode {index}'
    label = f'seed {seed}'
    if expert is not None and expert != seed:
        label += f' / expert {expert}'
    if start is not None and goal is not None:
        label += f' · steps {start}→{goal}'
    return label


def episode_results(
    evaluations: list[Evaluation],
) -> tuple[list[tuple], list[list[EpisodeResult | None]]]:
    """Align episodes across runs; rows are cases, columns are runs."""
    per_run = []
    for evaluation in evaluations:
        cases = episode_cases(evaluation)
        arrays = evaluation.metric_arrays
        outcomes = evaluation.episode_successes
        videos: dict[int, tuple[Path, bool | None]] = {}
        for path in sorted(evaluation.source.parent.glob('*.mp4')):
            match = VIDEO.match(path.name)
            if match:
                outcome = match[2] and match[2].lower() == 'true'
                videos[int(match[1])] = (path, outcome)
        results: dict[tuple, EpisodeResult] = {}
        for index, case in enumerate(cases):
            video, outcome = videos.get(index, (None, None))
            errors = {
                key: values[index]
                for key in ERROR_METRICS
                if len(values := arrays.get(f'episode_{key}', []))
                == len(cases)
                and math.isfinite(values[index])
            }
            results.setdefault(
                case,
                EpisodeResult(
                    outcomes[index] if index < len(outcomes) else outcome,
                    errors,
                    video,
                ),
            )
        per_run.append(results)
    cases = sorted(
        set().union(*per_run),
        key=lambda case: tuple(-math.inf if v is None else v for v in case),
    )
    return cases, [
        [results.get(case) for results in per_run] for case in cases
    ]


def outcome_group(results: list[EpisodeResult | None]) -> str:
    outcomes = {r.success for r in results if r and r.success is not None}
    if outcomes == {True, False}:
        return 'differs'
    return {frozenset({False}): 'failed', frozenset({True}): 'succeeded'}.get(
        frozenset(outcomes), 'unknown'
    )


OUTCOME_GROUPS = {
    'differs': 'Outcomes differ across runs',
    'failed': 'Failed in every run',
    'succeeded': 'Succeeded in every run',
    'unknown': 'Outcome not recorded',
}


def copy_asset(source: Path, target: Path, copy: bool) -> Path:
    if not copy:
        return source
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def outcome_mark(result: EpisodeResult | None) -> str:
    if result is None or result.success is None:
        return ''
    return '✓' if result.success else '✗'


def episode_lines(
    cases: list[tuple],
    rows: list[list[EpisodeResult | None]],
    output: Path,
    episodes_path: Path,
) -> list[str]:
    """Outcome matrix with one row per shared episode, grouped by agreement."""
    if not cases:
        return []
    groups: dict[str, list[int]] = {key: [] for key in OUTCOME_GROUPS}
    for index, results in enumerate(rows):
        groups[outcome_group(results)].append(index)
    runs = len(rows[0])
    summary = ', '.join(
        f'{len(groups[key])} {label[0].lower()}{label[1:]}'
        for key, label in OUTCOME_GROUPS.items()
        if groups[key]
    )
    lines = [
        '',
        '## Episode outcomes',
        '',
        (
            'Episodes are matched across runs by eval seed, expert seed and '
            'start/goal steps (by index when seeds are missing). ✓ success, '
            '✗ failure, blank means the run has no such episode. Click an '
            'episode for its videos from every run side by side.'
        ),
        '',
        f'{len(cases)} episodes: {summary}.',
    ]
    header = [
        'Episode',
        'Successes',
        *(f'Run {i}' for i in range(1, runs + 1)),
    ]
    for key, label in OUTCOME_GROUPS.items():
        if not groups[key]:
            continue
        table = [
            '| ' + ' | '.join(header) + ' |',
            '| --- | ---: |' + ' :---: |' * runs,
        ]
        for index in groups[key]:
            results = rows[index]
            known = [r for r in results if r and r.success is not None]
            link = f'{relative_url(episodes_path, output)}#episode-{index}'
            table.append(
                f'| [{cell(case_label(cases[index]))}]({link}) '
                f'| {sum(r.success for r in known)}/{len(known)} | '
                + ' | '.join(outcome_mark(r) for r in results)
                + ' |'
            )
        title = f'{label} ({len(groups[key])})'
        if key == 'succeeded':
            lines.extend(
                ['', '<details>', f'<summary>{title}</summary>', '', *table]
            )
            lines.extend(['', '</details>'])
        else:
            lines.extend(['', f'### {title}', '', *table])
    return lines


def write_episodes(
    cases: list[tuple],
    rows: list[list[EpisodeResult | None]],
    videos: list[list[Path | None]],
    path: Path,
    output: Path,
) -> None:
    """Show each episode's videos from every run side by side."""
    lines = [
        '# Episode videos',
        '',
        (
            'The same episode (seed and start/goal steps) from every run. '
            'Episodes that failed somewhere come first.'
        ),
        '',
        f'[Back to evaluation results]({relative_url(output, path)})',
    ]
    order = sorted(
        range(len(cases)),
        key=lambda i: list(OUTCOME_GROUPS).index(outcome_group(rows[i])),
    )
    for index in order:
        results = rows[index]
        known = [r for r in results if r and r.success is not None]
        lines.extend(
            [
                '',
                f'<a id="episode-{index}"></a>',
                '',
                f'## {cell(case_label(cases[index]))}',
                '',
                (
                    f'{OUTCOME_GROUPS[outcome_group(results)]}; '
                    f'{sum(r.success for r in known)}/{len(known)} runs '
                    'succeed.'
                ),
                '',
                '<table>',
            ]
        )
        entries = [
            (run, result, video)
            for run, (result, video) in enumerate(
                zip(results, videos[index]), 1
            )
            if result is not None
        ]
        for start in range(0, len(entries), 4):
            lines.append('<tr>')
            for run, result, video in entries[start : start + 4]:
                errors = '<br>'.join(
                    f'{ERROR_METRICS[key]}: {value:.4g}'
                    for key, value in result.errors.items()
                )
                player = (
                    '<video controls width="320" preload="none">'
                    f'<source src="{relative_url(video, path)}" '
                    'type="video/mp4"></video><br>'
                    if video
                    else '<i>No video</i><br>'
                )
                lines.append(
                    f'<td>{player}<b>Run {run}</b> {outcome_mark(result)}'
                    f'<br>{errors}</td>'
                )
            lines.append('</tr>')
        lines.append('</table>')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


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


def plain(value: Any) -> str:
    text = str(value)
    if isinstance(value, (str, Path)) and (
        text.startswith(('/', '~/')) or re.match(r'[A-Za-z]:[/\\]', text)
    ):
        text = short_path(text)
    return portable_text(text)


def cell(value: Any) -> str:
    return (
        plain(value)
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


FAILURE_LABELS = {
    'xy_error_m': 'T position',
    'yaw_error_rad': 'T orientation',
    'eef_position_error_m': 'EEF position',
}


def failed_checks(errors: dict, tolerances: list) -> str:
    """Name the goal tolerances an outcome's final errors exceed."""
    failed = [
        FAILURE_LABELS[key]
        for key, tolerance in tolerances
        if tolerance is not None
        and isinstance(errors.get(key), (int, float))
        and errors[key] > tolerance
    ]
    return ', '.join(failed) or '—'


def experiment_section(
    title: str,
    files: list[Path],
    output: Path,
    copy_assets: bool,
    warnings: list[str],
    landscape: bool = False,
    root: Path | None = None,
    include_videos: bool = True,
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

    def run_label(path: Path) -> str:
        # Diagnostic dirs share generic names, so show which run they belong to.
        if root is not None and path.parent.is_relative_to(root.resolve()):
            relative = path.parent.relative_to(root.resolve()).as_posix()
            if relative != '.':
                return relative
        return path.parent.name

    for (path, data), config in zip(rows, configs):
        values = [cell(run_label(path))]
        values.extend(cell(data.get(key, '—')) for key in metric_keys)
        values.extend(cell(config.get(key, '—')) for key in differing)
        lines.append('| ' + ' | '.join(values) + ' |')
    for index, (path, diagnostics) in enumerate(rows, 1):
        lines.extend(
            [
                '',
                f'### {cell(run_label(path))}',
                '',
            ]
        )
        world = diagnostics.get('config', {}).get('world', {}) or {}
        # Mirrors ManiSkill PushT goal success; EEF counts only when set.
        tolerances = [
            ('xy_error_m', world.get('goal_position_tolerance', 0.02)),
            ('yaw_error_rad', world.get('goal_yaw_tolerance', 0.15)),
            (
                'eef_position_error_m',
                world.get('goal_eef_position_tolerance'),
            ),
        ]
        if landscape:
            video = diagnostics.get('comparison_video')
            if include_videos and isinstance(video, str):
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
                for name in ('landscape.png', 'landscape_3d.png')
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
                if include_videos and copy_assets:
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
                lines.extend([f'**{cell(label)}**', ''])
                if include_videos:
                    lines.extend(
                        [
                            '<video controls width="960" preload="none">',
                            f'  <source src="{relative_url(target, output)}" type="video/mp4">',
                            '</video>',
                            '',
                        ]
                    )
                outcomes = comparison.get('outcomes', {})
                if isinstance(outcomes, dict):
                    lines.extend(
                        [
                            '| Sequence | Final goal success | T XY error (m) '
                            '| T yaw error (rad) | EEF position error (m) '
                            '| Failed on |',
                            '| --- | --- | --- | --- | --- | --- |',
                        ]
                    )
                    for name, outcome in outcomes.items():
                        if isinstance(outcome, dict):
                            errors = outcome.get('final_goal_errors') or {}
                            row = [
                                name,
                                outcome.get('final_goal_success', '—'),
                                *(
                                    round(errors[key], 4)
                                    if isinstance(errors.get(key), float)
                                    else errors.get(key, '—')
                                    for key, _ in tolerances
                                ),
                                failed_checks(errors, tolerances),
                            ]
                            lines.append(
                                '| ' + ' | '.join(map(cell, row)) + ' |'
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


def error_plots(
    evaluations: list[Evaluation], output: Path, root: Path
) -> list[str]:
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
            outside_legend(fig, len(evaluations))
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
            [str(index) for index in range(1, len(evaluations) + 1)],
            fontsize=8,
        )
        means_ax.set_xlim(-0.75, len(evaluations) - 0.25)
        means_ax.set_xlabel('Run (numbers match the results table)')
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
    lines.extend(
        mean_error_lines(evaluations, metrics, figure_dir, output, root)
    )
    return lines


def outside_legend(fig, entries: int, *handles_labels) -> None:
    """Legend right of the axes, in enough columns to fit the figure height."""
    rows = max(1, int(fig.get_figheight() / 0.18))
    fig.legend(
        *handles_labels,
        loc='outside right upper',
        ncol=max(1, math.ceil(entries / rows)),
        fontsize=8,
    )


def mean_error_lines(
    evaluations: list[Evaluation],
    metrics: list[str],
    figure_dir: Path,
    output: Path,
    root: Path,
) -> list[str]:
    """Plot success rate and mean errors as lines, one per run name with
    its _epN removed."""
    import matplotlib.pyplot as plt

    groups: dict[str, list[tuple[int, Evaluation]]] = {}
    epochs = True
    for index, evaluation in enumerate(evaluations, 1):
        group, epoch = run_group(evaluation.source.parent, root)
        epochs = epochs and epoch is not None
        groups.setdefault(group, []).append((epoch or index, evaluation))
    panels = [
        ('success_rate', 'Success rate (%)'),
        *((key, ERROR_METRICS[key]) for key in metrics),
    ]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9, 3.5 * len(panels)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (key, label) in zip(axes[:, 0], panels):
        for group, points in groups.items():
            points = sorted(
                (
                    x,
                    e.success_rate
                    if key == 'success_rate'
                    else e.mean_errors.get(key, math.nan),
                )
                for x, e in points
            )
            points = [(x, y) for x, y in points if math.isfinite(y)]
            if points:
                ax.plot(*zip(*points), marker='o', label=group)
        ax.set_ylabel(label)
        ax.set_xlabel('Epoch' if epochs else 'Run')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axes[0, 0].set_title('Success rate')
    axes[1, 0].set_title('Mean final error across episodes')
    if groups:
        # Every run has a success rate, so the top panel lists all groups.
        handles, labels = axes[0, 0].get_legend_handles_labels()
        outside_legend(fig, len(handles), handles, labels)
    target = figure_dir / 'mean_errors.png'
    try:
        fig.savefig(target, dpi=150)
    finally:
        plt.close(fig)
    return [
        f'![Success rate and mean errors]({relative_url(target, output)})',
        '',
    ]


def write_report(
    evaluations: list[Evaluation],
    output: Path,
    root: Path,
    prefix: str,
    copy_videos: bool = True,
    warnings: list[str] | None = None,
    landscape_dirs: list[Path] | None = None,
    expert_dirs: list[Path] | None = None,
    include_videos: bool = True,
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
    # Lowest finite mean per error column, underlined when runs compete.
    best_errors = {}
    for key in mean_keys:
        finite = [
            e.mean_errors[key]
            for e in evaluations
            if math.isfinite(e.mean_errors.get(key, math.nan))
        ]
        best_errors[key] = min(finite) if len(finite) > 1 else None

    def error_cell(evaluation, key):
        value = evaluation.mean_errors.get(key, math.nan)
        if not math.isfinite(value):
            return '—'
        text = f'{value:.6g}'
        return f'<u>{text}</u>' if value == best_errors[key] else text

    headers = [
        'Evaluation',
        'Success rate (%)',
        *(f'Mean {ERROR_METRICS[key]}' for key in mean_keys),
        *map(cell, differing),
    ]
    lines = [
        '# Evaluation results',
        '',
        (
            f'Evaluation directory names start with `{prefix}`.'
            if prefix
            else 'All evaluation directories under the root.'
        ),
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
            'below. A dash means the parameter was absent from that run. '
            'The lowest mean error in each column is underlined.'
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
            *(error_cell(evaluation, key) for key in mean_keys),
            *(cell(config.get(key, '—')) for key in differing),
        ]
        lines.append('| ' + ' | '.join(values) + ' |')

    lines.extend(error_plots(evaluations, output, root))

    cases, rows = episode_results(evaluations)
    episodes_path = output.with_name(f'{output.stem}_episodes.md')
    videos = [
        [
            copy_asset(
                result.video,
                output.parent
                / f'{output.stem}_videos'
                / f'run_{run:03d}'
                / result.video.name,
                copy_videos,
            )
            if include_videos and result and result.video
            else None
            for run, result in enumerate(results, 1)
        ]
        for results in rows
    ]
    lines.extend(episode_lines(cases, rows, output, episodes_path))
    if cases:
        write_episodes(cases, rows, videos, episodes_path, output)
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
    diagnostic_lines: list[str] = []
    diagnostic_shared: list[str] = []
    for title, files, landscape in (
        (
            'Action-space energy landscapes',
            experiment_files(landscape_dirs or [], 'results.json'),
            True,
        ),
        ('Expert-action selection', expert_files, False),
    ):
        section, common = experiment_section(
            title,
            files,
            output,
            copy_videos,
            warnings,
            landscape,
            root,
            include_videos,
        )
        lines.extend(section)
        shared_lines.extend(common)
        diagnostic_lines.extend(section)
        diagnostic_shared.extend(common)
    shared_path.write_text('\n'.join(shared_lines) + '\n', encoding='utf-8')
    if warnings:
        lines.extend(['', '## Skipped result files', ''])
        lines.extend(
            f'- {cell(portable_text(warning))}' for warning in warnings
        )
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    runs = []
    for index, (evaluation, config) in enumerate(zip(evaluations, configs), 1):
        directory = evaluation.source.parent
        group, epoch = run_group(directory, root)
        runs.append(
            {
                'id': index,
                'name': plain(directory.relative_to(root).as_posix()),
                'group': group,
                'epoch': epoch,
                'records': evaluation.record_count,
                'success_rate': evaluation.success_rate,
                'means': {
                    key: value
                    for key, value in evaluation.mean_errors.items()
                    if math.isfinite(value)
                },
                'params': {
                    key: plain(config.get(key, '—')) for key in differing
                },
            }
        )
    for run in runs:
        run['facets'] = run_facets(run['params'])
    # Keep only facets that distinguish runs.
    for name in {name for run in runs for name in run['facets']}:
        if len({run['facets'].get(name) for run in runs}) < 2:
            for run in runs:
                run['facets'].pop(name, None)
    html_path = output.with_suffix('.html')
    data = {
        'prefix': prefix,
        'metrics': ERROR_METRICS,
        'runs': runs,
        'differing': differing,
        'shared': {key: plain(value) for key, value in shared.items()},
        'episodes': [
            {
                'label': case_label(case),
                'case': dict(
                    zip(('seed', 'expert', 'start', 'goal', 'index'), case)
                ),
                'group': outcome_group(results),
                'results': [
                    result
                    and {
                        'success': result.success,
                        'errors': result.errors,
                        'video': video and relative_url(video, html_path),
                    }
                    for result, video in zip(results, episode_videos)
                ],
            }
            for case, results, episode_videos in zip(cases, rows, videos)
        ],
        'diagnostics': markdown_html(diagnostic_lines, output, html_path),
        'diagnostic_shared': markdown_html(
            diagnostic_shared, output, html_path
        ),
        'warnings': [portable_text(warning) for warning in warnings],
    }
    template = Path(__file__).with_name('eval_report.html')
    html_path.write_text(
        template.read_text(encoding='utf-8').replace(
            '__REPORT_DATA__',
            json.dumps(data, allow_nan=False).replace('</', '<\\/'),
        ),
        encoding='utf-8',
    )


def markdown_html(lines: list[str], source: Path, target: Path) -> str:
    """Convert the report's own Markdown subset (headings, tables, images)."""
    html: list[str] = []
    table: list[list[str]] = []

    def link(url: str) -> str:
        if '://' in url:
            return url
        return relative_url(source.parent / unquote(url), target)

    def flush() -> None:
        if table:
            header, _, *body = table
            html.append(
                '<div class="table-wrap"><table><thead><tr>'
                + ''.join(f'<th>{value}</th>' for value in header)
                + '</tr></thead><tbody>'
                + ''.join(
                    '<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>'
                    for row in body
                )
                + '</tbody></table></div>'
            )
            table.clear()

    for line in lines:
        if line.startswith('|'):
            table.append(
                [v.strip() for v in line.strip().strip('|').split('|')]
            )
            continue
        flush()
        if match := re.fullmatch(r'(#{2,3}) (.*)', line):
            level = len(match[1])
            html.append(f'<h{level}>{match[2]}</h{level}>')
        elif match := re.fullmatch(r'!\[(.*)\]\((.*)\)', line):
            html.append(f'<img alt="{match[1]}" src="{link(match[2])}">')
        elif match := re.search(r'src="([^"]+)"', line):
            html.append(line.replace(match[1], link(match[1])))
        elif line.startswith('**') and line.endswith('**'):
            html.append(f'<p><b>{line[2:-2]}</b></p>')
        elif line.startswith('<'):
            html.append(line)
        elif line:
            html.append(f'<p>{line}</p>')
    flush()
    return '\n'.join(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'root', type=Path, help='Directory searched recursively'
    )
    parser.add_argument(
        '--prefix',
        default='',
        help='Literal evaluation directory prefix (default: all directories)',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('results.md'),
        help='Report path; .md and .html are both written (default: results.md)',
    )
    parser.add_argument(
        '--result-pattern',
        default='*_results.txt',
        help='Result filename glob within each matched directory',
    )
    parser.add_argument(
        '--exclude',
        action='append',
        default=[],
        help=(
            'Skip directories whose path under the root matches this glob, '
            "e.g. '*/expert_candidates' (repeatable)"
        ),
    )
    parser.add_argument(
        '--link-videos',
        action='store_true',
        help='Link original videos and figures instead of copying them',
    )
    parser.add_argument(
        '--no-videos',
        dest='include_videos',
        action='store_false',
        help='Omit episode and diagnostic videos; keep tables and plots',
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
    evaluations: list[Evaluation] = []
    warnings: list[str] = []
    for path in discover(root, args.prefix, args.result_pattern, args.exclude):
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
        args.output.expanduser().resolve().with_suffix('.md'),
        root,
        args.prefix,
        copy_videos=not args.link_videos,
        warnings=warnings,
        landscape_dirs=args.landscape_dir,
        expert_dirs=args.expert_dir,
        include_videos=args.include_videos,
    )
    output = args.output.with_suffix('')
    print(
        f'Wrote {len(evaluations)} evaluations to {output}.md and {output}.html'
    )


if __name__ == '__main__':
    main()
