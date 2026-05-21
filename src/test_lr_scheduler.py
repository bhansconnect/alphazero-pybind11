"""Unit tests for PlateauLRScheduler."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lr_scheduler import PlateauLRScheduler, ema_update


def _make(threshold=0.002, patience=10, conv_patience=10, cooldown=0, max_drops=3):
    log = []

    def set_lr(lr):
        log.append(lr)

    sched = PlateauLRScheduler(
        set_lr=set_lr,
        initial_lr=1.0,
        drop_factor=0.5,
        max_drops=max_drops,
        patience=patience,
        conv_patience=conv_patience,
        cooldown=cooldown,
        threshold=threshold,
    )
    return sched, log


def test_monotone_improving_never_drops():
    sched, log = _make(patience=5)
    for i in range(100):
        ema = 10.0 * (0.95 ** i)
        event, _ = sched.step(ema)
        assert event is None
    assert sched.drops == 0
    assert log == [1.0]  # only the initial set_lr


def test_plateau_after_fall_drops_once_at_patience():
    sched, log = _make(patience=10)
    # Drop fast for 5 steps, then flatline at 1.0
    for v in [10.0, 5.0, 2.0, 1.5, 1.0]:
        sched.step(v)
    assert sched.drops == 0
    # Now flatline: should fire one drop after `patience+1` non-improving steps
    for _ in range(20):
        sched.step(1.0)
    assert sched.drops >= 1
    assert log[-1] == 0.5  # 1.0 * 0.5


def test_oscillation_within_prior_range_drops():
    """The original bug case: loss bouncing within an already-seen range."""
    sched, _ = _make(patience=20, threshold=0.005)
    # Descend to 1.0 first
    for v in [10.0, 5.0, 2.0, 1.5, 1.0]:
        sched.step(v)
    drops_before = sched.drops
    # Oscillate between 1.0 and 1.2 — never beats the lifetime best of 1.0 by threshold
    for i in range(100):
        sched.step(1.0 + 0.1 * (i % 2))
    assert sched.drops > drops_before, "Oscillation within prior range should trigger LR drop"


def test_cooldown_blocks_immediate_redrop():
    sched, log = _make(patience=5, cooldown=20)
    # Descend then flatline to force first drop
    sched.step(10.0)
    sched.step(1.0)
    for _ in range(10):
        sched.step(1.0)
    # First drop should have fired
    assert sched.drops == 1
    drop_step = len(log)  # log entries: initial + first drop = 2
    # Continue flatlining — cooldown should block re-drop for `cooldown` steps
    for _ in range(15):
        event, _ = sched.step(1.0)
        assert event != "drop", "Cooldown should block re-drop"
    # After cooldown elapses + patience, drop fires again
    for _ in range(30):
        sched.step(1.0)
    assert sched.drops >= 2


def test_micro_improvements_dont_reset_counter():
    """Improvements smaller than threshold should NOT reset the bad-step counter.

    Even cumulative improvements that never individually beat the lifetime best
    by the threshold leave the counter free to accumulate.
    """
    sched, _ = _make(patience=10, threshold=0.10)  # require 10% improvement vs best
    sched.step(10.0)  # best = 10
    # Hover near 10.0 with tiny noise — never beats 10 * 0.9 = 9.0
    for i in range(50):
        sched.step(10.0 - 0.001 * i)  # 9.95 floor after 50 steps, well above 9.0
    assert sched.drops >= 1, (
        f"Below-threshold drift should not prevent LR drop; "
        f"got drops={sched.drops}, num_bad={sched.num_bad}"
    )


def test_max_drops_then_stop():
    sched, _ = _make(patience=3, max_drops=2)
    sched.step(10.0)
    # Flatline to exhaust all drops + final stop
    last_event = None
    for _ in range(200):
        event, _ = sched.step(1.0)
        if event == "stop":
            last_event = "stop"
            break
        if event == "drop":
            last_event = "drop"
    assert sched.drops == 2
    assert last_event == "stop"


def test_ema_update_helper():
    assert ema_update(None, 5.0) == 5.0
    # beta=0.99 default
    assert abs(ema_update(10.0, 5.0) - (0.99 * 10.0 + 0.01 * 5.0)) < 1e-9
    # custom beta
    assert abs(ema_update(10.0, 5.0, beta=0.5) - 7.5) < 1e-9
