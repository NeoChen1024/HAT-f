import os
import sys
import unittest
from unittest import mock

import click
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import batch_infer


class NearestX4(torch.nn.Module):

    def forward(self, x):
        return x.repeat_interleave(batch_infer.SCALE, dim=2).repeat_interleave(batch_infer.SCALE, dim=3)


class BatchInferTest(unittest.TestCase):

    def test_tile_infer_matches_full_model_for_offset_sensitive_input(self):
        orig_zeros = torch.zeros

        def zeros_cpu(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            return orig_zeros(*args, **kwargs)

        lq = torch.arange(1 * 3 * 18 * 20, dtype=torch.float32).view(1, 3, 18, 20) / 255.0
        with mock.patch.object(torch.Tensor, "cuda", lambda self: self), mock.patch.object(batch_infer.torch, "zeros", zeros_cpu):
            tiled = batch_infer.tile_infer(NearestX4(), lq, tile_size=16, tile_pad=16)

        expected = NearestX4()(lq)
        torch.testing.assert_close(tiled, expected)

    def test_pad_to_window_handles_tiny_images(self):
        tiny = torch.ones(1, 3, 1, 1)

        padded = batch_infer.pad_to_window(tiny)

        self.assertEqual(padded.shape, (1, 3, batch_infer.WINDOW_SIZE, batch_infer.WINDOW_SIZE))

    def test_tensor_to_numpy_rounds_like_basicsr_tensor2img(self):
        tensor = torch.full((1, 3, 1, 1), 1.6 / 255.0)

        arr = batch_infer.tensor_to_numpy(tensor)

        self.assertEqual(arr.dtype, np.uint8)
        self.assertEqual(arr.tolist(), [[[2, 2, 2]]])

    def test_validate_multiple_of_16_rejects_zero(self):
        with self.assertRaises(click.BadParameter):
            batch_infer._validate_multiple_of_16(None, None, 0)

    def test_validate_multiple_of_16_rejects_negative(self):
        with self.assertRaises(click.BadParameter):
            batch_infer._validate_multiple_of_16(None, None, -16)

    def test_extract_state_dict_prefers_ema_and_strips_module_prefix(self):
        checkpoint = {
            "params_ema": {
                "module.conv.weight": torch.ones(1),
            },
            "params": {
                "module.conv.weight": torch.zeros(1),
            },
        }

        state = batch_infer.extract_state_dict(checkpoint)

        self.assertEqual(list(state), ["conv.weight"])
        torch.testing.assert_close(state["conv.weight"], torch.ones(1))


if __name__ == "__main__":
    unittest.main()
