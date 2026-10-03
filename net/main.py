"""Training entry point; supports both python net/main.py and python -m net.main."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn, optim

from net.data_utils import create_loaders
from net.engine import Trainer
from net.models import FFA, PerLoss
from net.option import parse_options


def main(argv=None):
    options = parse_options(argv)
    loader_train, loader_test = create_loaders(options)
    model = FFA(gps=options.gps, blocks=options.blocks).to(options.device)
    if options.device == "cuda":
        model = nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    criterion = [nn.L1Loss().to(options.device)]
    if options.perloss:
        from torchvision.models import VGG16_Weights, vgg16

        features = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(options.device)
        features.requires_grad_(False)
        features.eval()
        criterion.append(PerLoss(features).to(options.device))
    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=options.lr, betas=(0.9, 0.999), eps=1e-8,
    )
    print(options)
    return Trainer(model, optimizer, criterion, options).fit(loader_train, loader_test)


if __name__ == "__main__":
    main()
