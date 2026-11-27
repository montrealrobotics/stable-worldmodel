"""Re-render an OverheadPushT dataset's frames from its saved simulator states.

Every row of a dataset collected with collect_maniskill_pusht.py stores the
full restorable ManiSkill state (`state`). This script loads those states into
a batched GPU OverheadPushT-v1 env (no physics steps) and renders another
camera, by default ManiSkill's stock PushT `base_camera`. The output dataset
keeps every column and the row order of the source and only replaces
`pixels`; the `_episodes` side table and the collection JSON are copied (the
JSON gains a `rerender` entry).

Check: the first --verify-batches batches also compare the re-rendered source
camera (collection JSON `camera_name`, e.g. overhead_camera) with the source
pixels; mean abs difference ~1/255 is JPEG noise, more means the states do not
restore the scene.

Example:
    python scripts/data/rerender_pusht_dataset.py \
        --source $SCRATCH/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance \
        --output $SCRATCH/le-wm/datasets/basecam_fov1_noisy_replay_100k.lance \
        --camera-fov 1.0
"""

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--camera-name', default='base_camera')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument(
        '--camera-fov',
        type=float,
        default=None,
        help='Radians; base_camera default pi/2, ManiSkill videos use 1.0.',
    )
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--verify-batches', type=int, default=2)
    p.add_argument('--limit', type=int, default=None, help='Rows (for tests).')
    p.add_argument('--control-mode', default='pd_ee_delta_xy')
    p.add_argument('--encode-threads', type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    import gymnasium as gym
    import lance
    import pyarrow as pa
    import torch
    from PIL import Image

    from stable_worldmodel.data.formats.lance import _encode_frame
    from stable_worldmodel.envs.maniskill import overhead_pusht  # noqa: F401

    if args.output.exists():
        raise FileExistsError(args.output)
    src = lance.dataset(str(args.source))
    schema = src.schema
    if 'state' not in schema.names or 'pixels' not in schema.names:
        raise ValueError('source needs `state` and `pixels` columns')
    total = src.count_rows() if args.limit is None else args.limit
    other = [n for n in schema.names if n != 'pixels']

    meta_src = args.source.with_suffix('.collection.json')
    meta = json.loads(meta_src.read_text()) if meta_src.exists() else {}
    # Verify against the camera the source saved, when it is rendered as-is.
    source_camera = meta.get('camera_name', 'overhead_camera')
    if source_camera == args.camera_name:
        source_camera = None

    camera = {'width': args.image_size, 'height': args.image_size}
    if args.camera_fov is not None:
        camera['fov'] = args.camera_fov
    env = gym.make(
        'OverheadPushT-v1',
        num_envs=args.num_envs,
        obs_mode='rgb',
        render_mode='rgb_array',
        sim_backend='physx_cuda',
        control_mode=args.control_mode,
        include_base_camera=args.camera_name == 'base_camera',
        sensor_configs={args.camera_name: camera},
    ).unwrapped
    env.reset(seed=0)
    state_size = env.get_state().shape[-1]
    if state_size != schema.field('state').type.list_size:
        raise ValueError(
            f'env state size {state_size} != dataset '
            f'{schema.field("state").type.list_size}'
        )

    def render(states):
        n = len(states)
        batch = torch.as_tensor(states, device=env.device)
        if n < args.num_envs:  # pad the last batch with copies
            pad = batch[-1:].expand(args.num_envs - n, -1)
            batch = torch.cat([batch, pad])
        env.set_state(batch)
        # Push poses/joints to PhysX, refresh link poses from qpos, read back.
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()
        sensors = env._get_obs_sensor_data()
        out = {k: v['rgb'][:n].cpu().numpy() for k, v in sensors.items()}
        return out

    encoder = ThreadPoolExecutor(args.encode_threads)
    stats = {'rows': 0, 'verify_mae': []}
    started = time.perf_counter()

    def batches():
        done = 0
        for index, rb in enumerate(
            src.to_batches(columns=other, batch_size=args.num_envs)
        ):
            if done >= total:
                break
            rb = rb.slice(0, min(rb.num_rows, total - done))
            states = (
                rb.column('state')
                .flatten()
                .to_numpy()
                .reshape(rb.num_rows, -1)
            )
            frames = render(states.astype(np.float32))
            if index < args.verify_batches and source_camera in frames:
                ref = src.take(
                    list(range(done, done + rb.num_rows)), columns=['pixels']
                ).column('pixels')
                saved = np.stack(
                    [
                        np.array(
                            Image.open(io.BytesIO(b.as_py())).convert('RGB')
                        )
                        for b in ref
                    ]
                )
                mae = float(
                    np.abs(
                        frames[source_camera].astype(int) - saved.astype(int)
                    ).mean()
                )
                stats['verify_mae'].append(mae)
                print(
                    f'verify batch {index}: {source_camera} re-render MAE '
                    f'{mae:.3f} (JPEG noise ~1)',
                    flush=True,
                )
            jpegs = list(
                encoder.map(
                    lambda f: _encode_frame(f, args.jpeg_quality),
                    frames[args.camera_name],
                )
            )
            columns = {n: rb.column(n) for n in other}
            columns['pixels'] = pa.array(jpegs, type=pa.binary())
            done += rb.num_rows
            stats['rows'] = done
            if index % 50 == 0:
                rate = done / (time.perf_counter() - started)
                print(
                    f'{done}/{total} rows, {rate:.0f} rows/s, '
                    f'ETA {(total - done) / rate / 3600:.2f}h',
                    flush=True,
                )
            yield pa.record_batch(
                [columns[n] for n in schema.names], schema=schema
            )

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    lance.write_dataset(reader, str(args.output), schema=schema, mode='create')

    episodes = args.source.with_name(args.source.stem + '_episodes.lance')
    if episodes.exists():
        ep = lance.dataset(str(episodes)).to_table()
        if args.limit is not None:  # keep only episodes present in the output
            import pyarrow.compute as pc

            present = pa.array(
                np.unique(
                    lance.dataset(str(args.output))
                    .to_table(columns=['episode_idx'])
                    .column('episode_idx')
                    .to_numpy()
                )
            )
            ep = ep.filter(
                pc.is_in(ep.column('episode_idx'), value_set=present)
            )
        lance.write_dataset(
            ep,
            str(args.output.with_name(args.output.stem + '_episodes.lance')),
            mode='create',
        )
    meta['rerender'] = {
        'source': str(args.source),
        'camera_name': args.camera_name,
        'image_size': args.image_size,
        'camera_fov': args.camera_fov,
        'jpeg_quality': args.jpeg_quality,
        'rows': stats['rows'],
        'verify_camera': source_camera,
        'verify_mae': stats['verify_mae'],
        'seconds': time.perf_counter() - started,
    }
    meta['camera_name'] = args.camera_name
    meta['image_size'] = args.image_size
    args.output.with_suffix('.collection.json').write_text(
        json.dumps(meta, indent=2) + '\n'
    )
    print(f'Wrote {stats["rows"]} rows to {args.output}')


if __name__ == '__main__':
    main()
