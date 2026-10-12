"""Batched Lance reads for current observations and sampled goals."""

import numpy as np
import pyarrow as pa
import torch

from .formats.lance import LanceDataset


class BatchedGoalLanceDataset(LanceDataset):
    """Fetch current clips and single-frame goals together for GoalDataset."""

    def _load_slices(self, slices: list[tuple[int, int, int]]) -> list[dict]:
        """Batch raw windows and single-frame goals through the same reader."""
        all_rows: list[int] = []
        row_offsets: list[int] = []
        sample_meta: list[tuple[int, int, int]] = []
        for ep_idx, start, end in slices:
            g_start = int(self.offsets[ep_idx] + start)
            g_end = int(self.offsets[ep_idx] + end)
            row_offsets.append(len(all_rows))
            all_rows.extend(range(g_start, g_end))
            sample_meta.append((ep_idx, g_start, g_end))

        big_batch = None
        image_pos: dict[int, int] = {}
        decoded_images: dict[str, torch.Tensor] = {}
        if self._fetch_columns and all_rows:
            unique_rows = sorted(set(all_rows))
            unique_batch = self._fetch_rows(unique_rows)
            unique_pos = {row: i for i, row in enumerate(unique_rows)}
            if len(unique_rows) == len(all_rows) and all_rows == unique_rows:
                big_batch = unique_batch
            else:
                gather = pa.array(
                    [unique_pos[r] for r in all_rows], type=pa.int64()
                )
                big_batch = unique_batch.take(gather)

            # Decode only observed frames, excluding skipped action rows.
            image_rows = sorted(
                {
                    row
                    for _, start, end in sample_meta
                    for row in range(start, end, self.frameskip)
                }
            )
            image_pos = {row: i for i, row in enumerate(image_rows)}
            image_batch = unique_batch.take(
                pa.array(
                    [unique_pos[row] for row in image_rows], type=pa.int64()
                )
            )
            for col in self.image_columns:
                if col in self._cache:
                    continue
                blobs = self._extract_column(image_batch, col)
                if isinstance(blobs, np.ndarray):
                    blobs = blobs.tolist()
                decoded_images[col] = self._decode_images(blobs)

        results: list[dict] = []
        for i, (ep_idx, g_start, g_end) in enumerate(sample_meta):
            sub_batch = (
                big_batch.slice(row_offsets[i], g_end - g_start)
                if big_batch is not None
                else None
            )
            if decoded_images:
                window_rows = range(g_start, g_end, self.frameskip)
                gather_idx = [image_pos[r] for r in window_rows]
                steps = self._process_batch(
                    ep_idx,
                    g_start,
                    sub_batch,
                    g_end=g_end,
                    decoded_images={
                        col: frames[gather_idx]
                        for col, frames in decoded_images.items()
                    },
                )
            else:
                steps = self._process_batch(
                    ep_idx, g_start, sub_batch, g_end=g_end
                )
            if self.transform:
                steps = self.transform(steps)
            results.append(steps)
        return results
