"""Shared LR scheduling utilities for inter-step EMA loss tracking."""

from typing import Callable, Optional, Tuple


def ema_update(prev: Optional[float], cur: float, beta: float = 0.99) -> float:
    """Exponential moving average update."""
    return cur if prev is None else beta * prev + (1.0 - beta) * cur


class PlateauLRScheduler:
    """PyTorch ReduceLROnPlateau-style LR drop on inter-step EMA loss.

    Tracks lifetime-best EMA. Counter increments on every non-improvement;
    resets only on a genuine new best beating ``threshold`` (relative). After
    ``patience`` bad steps fires a drop. Cooldown after each drop holds the
    counter at zero while the new LR settles.

    .step(ema) returns:
      - ("drop", new_lr) when a drop fires
      - ("stop", current_lr) when patience expires after max_drops
      - (None, current_lr) otherwise
    """

    def __init__(
        self,
        set_lr: Callable[[float], None],
        initial_lr: float,
        drop_factor: float,
        max_drops: int,
        patience: int,
        conv_patience: int,
        cooldown: int,
        threshold: float,
    ):
        self.set_lr = set_lr
        self.lr = initial_lr
        self.drop_factor = drop_factor
        self.max_drops = max_drops
        self.patience = patience
        self.conv_patience = conv_patience
        self.cooldown_steps = cooldown
        self.threshold = threshold
        self.best = float("inf")
        self.num_bad = 0
        self.cooldown_left = 0
        self.drops = 0
        self.final_dropped = False
        self.set_lr(self.lr)

    def step(self, ema_loss: float) -> Tuple[Optional[str], float]:
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            self.num_bad = 0
        elif ema_loss < self.best * (1.0 - self.threshold):
            self.best = ema_loss
            self.num_bad = 0
        else:
            self.num_bad += 1

        patience = self.conv_patience if self.final_dropped else self.patience
        if self.num_bad > patience:
            if not self.final_dropped and self.drops < self.max_drops:
                self.lr *= self.drop_factor
                self.drops += 1
                self.set_lr(self.lr)
                self.num_bad = 0
                self.cooldown_left = self.cooldown_steps
                if self.drops >= self.max_drops:
                    self.final_dropped = True
                return ("drop", self.lr)
            return ("stop", self.lr)
        return (None, self.lr)
