"""Behavioral regression checks using tiny local images; no dataset/download required."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from net.data_utils import HAZE_MEAN, HAZE_STD, Haze4K_Dataset, RESIDE_Dataset, create_loaders
from net.engine import Trainer
from net.evaluate_samples import evaluate
from net.models import FFA
from net.option import parse_options
from net.runtime import PROJECT_ROOT, load_checkpoint, resolve_path, save_checkpoint


def write_pair(root, haze_dir="haze", clear_dir="gt", extension=".png"):
    (root / haze_dir).mkdir(parents=True)
    (root / clear_dir).mkdir(parents=True)
    pixels = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    Image.fromarray(pixels).save(root / haze_dir / ("1_0.8_1.2" + extension))
    Image.fromarray(pixels).save(root / clear_dir / ("1" + extension))


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_imports_do_not_parse_arguments_or_create_outputs(self):
        code = (
            f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
            "sys.argv = ['consumer', '--unrelated-flag']; "
            "import net.option, net.data_utils, net.main, net.test, net.evaluate_samples"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_explicit_cpu_boolean_and_checkpoint_paths(self):
        options = parse_options(["--device", "cpu", "--resume", "False", "--model_dir", str(self.root / "nested")])
        self.assertEqual(options.device, "cpu")
        self.assertFalse(options.resume)
        self.assertEqual(options.model_dir, self.root / "nested" / "haze4k_train_ffa_3_20.pk")
        self.assertFalse(options.model_dir.parent.exists())
        explicit = parse_options(["--device", "cpu", "--no_resume", "--model_dir", str(self.root / "custom.pk")])
        self.assertEqual(explicit.model_dir.name, "custom.pk")
        self.assertEqual(resolve_path("data"), PROJECT_ROOT / "data")

    def test_invalid_training_parameters_fail_early(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_options(["--bs", "0"])

    def test_haze4k_small_images_stay_paired_during_augmentation(self):
        write_pair(self.root)
        (self.root / "haze" / "ignored.txt").write_text("not an image")
        dataset = Haze4K_Dataset(self.root, train=True, size=16)
        inputs, targets = dataset[0]
        self.assertEqual(tuple(inputs.shape), (3, 16, 16))
        mean = torch.tensor(HAZE_MEAN).view(3, 1, 1)
        std = torch.tensor(HAZE_STD).view(3, 1, 1)
        torch.testing.assert_close(inputs * std + mean, targets)
        self.assertEqual(len(dataset), 1)

    def test_reside_pairing_and_full_resolution_validation(self):
        write_pair(self.root, "hazy", "clear")
        dataset = RESIDE_Dataset(self.root, train=False, size="whole_img")
        self.assertEqual(tuple(dataset[0][1].shape), (3, 8, 12))

    def test_missing_ground_truth_reports_image(self):
        (self.root / "haze").mkdir()
        (self.root / "gt").mkdir()
        Image.new("RGB", (16, 16)).save(self.root / "haze" / "9_haze.png")
        with self.assertRaisesRegex(FileNotFoundError, "9_haze.png"):
            Haze4K_Dataset(self.root, train=True)

    def test_only_selected_dataset_is_constructed(self):
        write_pair(self.root / "Haze4K" / "train")
        write_pair(self.root / "Haze4K" / "test")
        options = SimpleNamespace(
            data_root=self.root, trainset="haze4k_train", testset="haze4k_test",
            bs=1, crop=True, crop_size=8, num_workers=0,
        )
        train, validation = create_loaders(options)
        self.assertEqual(len(train.dataset), 1)
        self.assertEqual(tuple(next(iter(validation))[0].shape), (1, 3, 8, 12))

    def test_legacy_dataparallel_checkpoint_and_plain_weights(self):
        model = FFA(gps=3, blocks=1)
        path = self.root / "legacy.pk"
        state = {"module." + key: value for key, value in model.state_dict().items()}
        torch.save({"model": state, "step": 7, "max_psnr": np.float64(20)}, path)
        target = FFA(gps=3, blocks=1)
        metadata = load_checkpoint(path, target)
        self.assertEqual(metadata["step"], 7)
        inputs = torch.rand(1, 3, 8, 8)
        with torch.inference_mode():
            torch.testing.assert_close(model(inputs), target(inputs))
        torch.save(model.state_dict(), self.root / "plain.pt")
        load_checkpoint(self.root / "plain.pt", nn.DataParallel(target))

    def test_checkpoint_restores_optimizer_state(self):
        model = nn.Conv2d(3, 3, 1)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.rand(1, 3, 8, 8)).sum().backward()
        optimizer.step()
        path = self.root / "nested" / "checkpoint.pk"
        save_checkpoint(path, nn.DataParallel(model), optimizer, step=1)
        target = nn.Conv2d(3, 3, 1)
        new_optimizer = torch.optim.Adam(target.parameters())
        load_checkpoint(path, target, optimizer=new_optimizer)
        self.assertEqual(len(new_optimizer.state), len(optimizer.state))
        self.assertEqual(int(next(iter(new_optimizer.state.values()))["step"]), 1)

    def test_training_consumes_batches_and_resumes_latest_progress(self):
        class RecordingModel(nn.Conv2d):
            def __init__(self):
                super().__init__(3, 3, 1)
                self.seen = []

            def forward(self, inputs):
                if self.training:
                    self.seen.append(round(inputs.mean().item(), 1))
                return super().forward(inputs)

        inputs = torch.stack([torch.full((3, 8, 8), value) for value in (0.2, 0.7)])
        loader = DataLoader(TensorDataset(inputs, inputs), batch_size=1, shuffle=False)
        options = SimpleNamespace(
            resume=False, model_dir=self.root / "model.pk", device="cpu",
            steps=2, lr=0.0001, no_lr_sche=True, perloss=False,
            eval_step=1, history_dir=self.root / "history", model_name="smoke",
        )
        model = RecordingModel()
        trainer = Trainer(model, torch.optim.Adam(model.parameters()), [nn.L1Loss()], options)
        with contextlib.redirect_stdout(io.StringIO()):
            result = trainer.fit(loader, loader)
        self.assertEqual(model.seen, [0.2, 0.7])
        self.assertEqual(result["step"], 2)
        self.assertEqual(len(result["ssims"]), 2)
        self.assertTrue(trainer.latest_path.is_file())
        options.resume, options.steps = True, 3
        resumed_model = RecordingModel()
        optimizer = torch.optim.Adam(resumed_model.parameters())
        resumed = Trainer(resumed_model, optimizer, [nn.L1Loss()], options)
        with contextlib.redirect_stdout(io.StringIO()):
            result = resumed.fit(loader, loader)
        self.assertEqual(result["step"], 3)
        self.assertEqual(len(result["losses"]), 3)
        self.assertEqual(int(next(iter(optimizer.state.values()))["step"]), 3)

    def test_saved_prediction_evaluation_uses_original_metrics(self):
        pred_dir, gt_dir = self.root / "pred", self.root / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()
        image = Image.new("RGB", (16, 16), color=(100, 120, 140))
        image.save(pred_dir / "1_haze_FFA.png")
        image.save(gt_dir / "1.png")
        rows, missing = evaluate(pred_dir, gt_dir)
        self.assertEqual(missing, [])
        self.assertEqual(rows[0]["psnr"], 100)
        self.assertAlmostEqual(rows[0]["ssim"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
