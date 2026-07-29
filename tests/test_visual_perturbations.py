from __future__ import annotations

import unittest

import torch

from codewam.codebook_eval.visual_perturbations import (
    ENDPOINT_FRAME_START,
    ENDPOINT_FRAME_STOP,
    RGB_CONDITIONS,
    _descriptor,
    _preprocess_resized_video,
    _prefix_change,
    _quantized_prefixes,
    _resize_float,
    _resize_uint8,
    _scale,
    _translate,
    apply_rgb_condition,
)
from codewam.codebook_eval.wan_probe_export import _preprocess_video


def condition(name: str):
    return next(value for value in RGB_CONDITIONS if value.name == name)


class VisualPerturbationTests(unittest.TestCase):
    def test_endpoint_condition_changes_only_current_temporal_block(self) -> None:
        frames = torch.full((45, 16, 16, 3), 100, dtype=torch.uint8)
        changed = apply_rgb_condition(
            frames,
            condition("endpoint_scale_090"),
        )

        torch.testing.assert_close(
            changed[:ENDPOINT_FRAME_START],
            frames[:ENDPOINT_FRAME_START],
        )
        torch.testing.assert_close(
            changed[ENDPOINT_FRAME_STOP:],
            frames[ENDPOINT_FRAME_STOP:],
        )
        self.assertEqual(tuple(changed.shape), tuple(frames.shape))

    def test_translate_and_scale_preserve_shape_without_wraparound(self) -> None:
        frames = torch.zeros((1, 5, 5, 3), dtype=torch.uint8)
        frames[:, 2, 2] = 255
        translated = _translate(frames, dx=1)
        scaled = _scale(frames, 1.2)

        self.assertEqual(tuple(translated.shape), tuple(frames.shape))
        self.assertEqual(int(translated[0, 2, 3, 0]), 255)
        self.assertEqual(int(translated[0, 2, 1, 0]), 0)
        self.assertEqual(tuple(scaled.shape), tuple(frames.shape))

    def test_descriptor_uses_exact_causal_indices(self) -> None:
        pooled = torch.arange(12, dtype=torch.float32).reshape(
            1, 12, 1, 1, 1
        )

        vector = _descriptor(pooled, stride=3, time_index=10)

        torch.testing.assert_close(
            vector,
            torch.tensor([[4.0, 7.0, 10.0]]),
        )

    def test_prefix_change_is_cumulative(self) -> None:
        level, prefix = _prefix_change(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 4, 3]]),
        )

        self.assertEqual(level, [False, True, False])
        self.assertEqual(prefix, [False, True, True])

    def test_quantized_prefixes_accumulate_rq_centers(self) -> None:
        codes = torch.tensor([[0, 1], [1, 0]])
        centers = (
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
            torch.tensor([[0.5, 0.5], [-1.0, 1.0]]),
        )

        prefixes = _quantized_prefixes(codes, centers)

        torch.testing.assert_close(
            prefixes[0],
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        )
        torch.testing.assert_close(
            prefixes[1],
            torch.tensor([[0.0, 1.0], [0.5, 2.5]]),
        )

    def test_resize_returns_uint8_target_shape(self) -> None:
        frames = torch.arange(
            2 * 8 * 12 * 3,
            dtype=torch.int32,
        ).remainder(256).to(torch.uint8).reshape(2, 8, 12, 3)

        resized = _resize_uint8(frames, 16, 32)

        self.assertEqual(tuple(resized.shape), (2, 16, 32, 3))
        self.assertEqual(resized.dtype, torch.uint8)

    def test_float_resize_preserves_canonical_identity_preprocessing(
        self,
    ) -> None:
        frames = torch.arange(
            5 * 18 * 30 * 3,
            dtype=torch.int32,
        ).remainder(256).to(torch.uint8).reshape(5, 18, 30, 3)

        expected = _preprocess_video(
            frames,
            height=16,
            width=32,
            dtype=torch.float32,
        )
        resized = _resize_float(frames, 16, 32)
        observed = _preprocess_resized_video(
            apply_rgb_condition(resized, condition("identity")),
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            observed,
            expected,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
