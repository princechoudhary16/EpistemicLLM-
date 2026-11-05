import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from confidence_calibration import config, modeling


class LoadModelPrecision(unittest.TestCase):
    def _loaded_dtype(self, device):
        with mock.patch.object(modeling, "AutoTokenizer"), \
                mock.patch.object(modeling, "AutoModelForCausalLM") as model_class:
            modeling.load_model(config.MODEL_ID, config.MODEL_REVISION, device)
        return model_class.from_pretrained.call_args.kwargs["dtype"]

    def test_every_device_uses_the_team_precision(self):
        for device in ("cuda", "mps", "cpu"):
            with self.subTest(device=device):
                self.assertIs(self._loaded_dtype(device), torch.float32)

    def test_pin_and_precision_come_from_the_shared_config(self):
        shared = yaml.safe_load((Path(config.CONFIG_DIR) / "model.yaml").read_text(encoding="utf-8"))
        self.assertEqual((config.MODEL_ID, config.MODEL_REVISION, config.TORCH_DTYPE),
                         (shared["model_id"], shared["revision"], shared["torch_dtype"]))


if __name__ == "__main__":
    unittest.main()
