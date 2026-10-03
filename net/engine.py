"""Training and validation, independent of CLI parsing and dataset discovery."""

import math
import time
from pathlib import Path

import numpy as np
import torch

from net.metrics import psnr, ssim
from net.runtime import load_checkpoint, save_checkpoint


def lr_schedule_cosdecay(step, total_steps, init_lr):
    return 0.5 * (1 + math.cos(step * math.pi / total_steps)) * init_lr


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    ssims, psnrs = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        prediction = model(inputs)
        ssims.append(ssim(prediction, targets).item())
        psnrs.append(psnr(prediction, targets))
    if not ssims:
        raise ValueError("Validation loader is empty")
    return float(np.mean(ssims)), float(np.mean(psnrs))


class Trainer:
    def __init__(self, model, optimizer, criterion, options):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.options = options
        self.state = dict(step=0, max_ssim=0.0, max_psnr=0.0, losses=[], ssims=[], psnrs=[])

    @property
    def latest_path(self):
        path = Path(self.options.model_dir)
        return path.with_name(f"{path.stem}_last{path.suffix}")

    def resume(self):
        if not self.options.resume:
            return
        # Prefer a complete latest checkpoint, falling back to legacy best-only files.
        path = self.latest_path if self.latest_path.is_file() else Path(self.options.model_dir)
        if path.is_file():
            checkpoint = load_checkpoint(path, self.model, self.options.device, self.optimizer)
            for key in self.state:
                if key in checkpoint:
                    self.state[key] = checkpoint[key]
            print(f"Resumed from {path} at step {self.state['step']}")

    def save(self, path):
        save_checkpoint(path, self.model, self.optimizer, **self.state)

    def fit(self, loader_train, loader_test):
        self.resume()
        started = time.monotonic()
        batches = iter(loader_train)
        for step in range(self.state["step"] + 1, self.options.steps + 1):
            self.model.train()
            try:
                inputs, targets = next(batches)
            except StopIteration:
                batches = iter(loader_train)
                try:
                    inputs, targets = next(batches)
                except StopIteration as error:
                    raise ValueError("Training loader is empty") from error
            learning_rate = self.options.lr
            if not self.options.no_lr_sche:
                learning_rate = lr_schedule_cosdecay(step, self.options.steps, self.options.lr)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            inputs, targets = inputs.to(self.options.device), targets.to(self.options.device)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(inputs)
            loss = self.criterion[0](prediction, targets)
            if self.options.perloss:
                loss = loss + 0.04 * self.criterion[1](prediction, targets)
            loss.backward()
            self.optimizer.step()
            self.state["step"] = step
            self.state["losses"].append(loss.item())
            elapsed = (time.monotonic() - started) / 60
            print(f"\rtrain loss: {loss.item():.5f} | step: {step}/{self.options.steps} | "
                  f"lr: {learning_rate:.7f} | minutes: {elapsed:.1f}", end="", flush=True)
            if step % self.options.eval_step == 0:
                ssim_value, psnr_value = evaluate(self.model, loader_test, self.options.device)
                self.state["ssims"].append(ssim_value)
                self.state["psnrs"].append(psnr_value)
                print(f"\nstep: {step} | ssim: {ssim_value:.4f} | psnr: {psnr_value:.4f}")
                # Preserve the original rule: both metrics must improve for best.
                if ssim_value > self.state["max_ssim"] and psnr_value > self.state["max_psnr"]:
                    self.state["max_ssim"] = ssim_value
                    self.state["max_psnr"] = psnr_value
                    self.save(self.options.model_dir)
                    print(f"Best model saved: {self.options.model_dir}")
                self.save(self.latest_path)
        # Always retain progress, even for runs shorter than eval_step.
        self.save(self.latest_path)
        history_dir = Path(self.options.history_dir)
        history_dir.mkdir(parents=True, exist_ok=True)
        for key in ("losses", "ssims", "psnrs"):
            np.save(history_dir / f"{self.options.model_name}_{self.state['step']}_{key}.npy", self.state[key])
        print()
        return self.state
