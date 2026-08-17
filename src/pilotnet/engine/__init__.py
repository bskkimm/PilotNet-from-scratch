"""Training and evaluation routines."""

from .trainer import evaluate, evaluate_temporal, train_epoch, train_temporal_epoch

__all__ = ["evaluate", "evaluate_temporal", "train_epoch", "train_temporal_epoch"]
