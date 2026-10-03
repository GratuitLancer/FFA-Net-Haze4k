"""Original FFA-Net model and perceptual loss exports."""

from .FFA import FFA
from .PerceptualLoss import LossNetwork as PerLoss

__all__ = ["FFA", "PerLoss"]
