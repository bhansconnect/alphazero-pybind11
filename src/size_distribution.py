"""Size distribution and curriculum learning for multi-size Star Gambit training.

This module manages the probability distribution over game sizes (Skirmish, Clash, Battle)
during training, implementing curriculum learning that starts with simpler games and
gradually transitions to an even distribution.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import alphazero


# Default curriculum: 80/15/5 -> 33/33/33 over 200 iterations
DEFAULT_CURRICULUM = [
    (0,   {'skirmish': 0.80, 'clash': 0.15, 'battle': 0.05}),
    (200, {'skirmish': 0.33, 'clash': 0.33, 'battle': 0.34}),
]

# Alternative curricula for experimentation
FAST_CURRICULUM = [
    (0,   {'skirmish': 0.60, 'clash': 0.25, 'battle': 0.15}),
    (50,  {'skirmish': 0.33, 'clash': 0.33, 'battle': 0.34}),
]

EVEN_FROM_START = [
    (0,   {'skirmish': 0.33, 'clash': 0.33, 'battle': 0.34}),
]

SKIRMISH_ONLY = [
    (0,   {'skirmish': 1.0, 'clash': 0.0, 'battle': 0.0}),
]


class SizeDistribution:
    """Manages the size distribution for multi-size training with curriculum learning.

    The distribution interpolates linearly between milestones, allowing smooth
    transitions from simple to complex games during training.

    Usage:
        dist = SizeDistribution()  # Uses default curriculum
        dist = SizeDistribution(milestones=FAST_CURRICULUM)  # Custom curriculum

        # During training:
        dist.step(iteration)  # Update distribution based on current iteration
        variant = dist.sample()  # Sample a game variant
    """

    VARIANT_NAMES = ['skirmish', 'clash', 'battle']
    VARIANT_MAP = {
        'skirmish': alphazero.StarGambitVariant.SKIRMISH,
        'clash': alphazero.StarGambitVariant.CLASH,
        'battle': alphazero.StarGambitVariant.BATTLE,
    }

    def __init__(self, milestones: Optional[List[Tuple[int, Dict[str, float]]]] = None):
        """Initialize size distribution with curriculum milestones.

        Args:
            milestones: List of (iteration, distribution) tuples defining the curriculum.
                        Each distribution is a dict mapping variant names to probabilities.
                        If None, uses DEFAULT_CURRICULUM.
        """
        self.milestones = milestones if milestones is not None else DEFAULT_CURRICULUM
        self._validate_milestones()

        # Current state
        self.current_iteration = 0
        self.current_probs = self._compute_probs(0)

    def _validate_milestones(self):
        """Validate that milestones are properly formatted."""
        if not self.milestones:
            raise ValueError("At least one milestone is required")

        prev_iter = -1
        for iteration, dist in self.milestones:
            if iteration <= prev_iter:
                raise ValueError(f"Milestones must be in increasing iteration order")
            prev_iter = iteration

            total = sum(dist.get(v, 0.0) for v in self.VARIANT_NAMES)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Distribution at iteration {iteration} must sum to 1, got {total}")

    def _compute_probs(self, iteration: int) -> np.ndarray:
        """Compute probability distribution for given iteration using linear interpolation."""
        # Find surrounding milestones
        prev_milestone = None
        next_milestone = None

        for i, (iter_val, dist) in enumerate(self.milestones):
            if iter_val <= iteration:
                prev_milestone = (iter_val, dist)
            if iter_val > iteration and next_milestone is None:
                next_milestone = (iter_val, dist)
                break

        # Before first milestone or only one milestone: use first milestone
        if prev_milestone is None:
            prev_milestone = self.milestones[0]

        # After last milestone: use last milestone
        if next_milestone is None:
            _, dist = prev_milestone
            return np.array([dist.get(v, 0.0) for v in self.VARIANT_NAMES])

        # Interpolate between milestones
        prev_iter, prev_dist = prev_milestone
        next_iter, next_dist = next_milestone

        alpha = (iteration - prev_iter) / (next_iter - prev_iter)

        probs = []
        for v in self.VARIANT_NAMES:
            p0 = prev_dist.get(v, 0.0)
            p1 = next_dist.get(v, 0.0)
            probs.append(p0 + alpha * (p1 - p0))

        return np.array(probs)

    def step(self, iteration: int) -> None:
        """Update distribution based on current iteration."""
        self.current_iteration = iteration
        self.current_probs = self._compute_probs(iteration)

    def sample(self) -> alphazero.StarGambitVariant:
        """Sample a variant based on current distribution."""
        idx = np.random.choice(3, p=self.current_probs)
        return self.VARIANT_MAP[self.VARIANT_NAMES[idx]]

    def sample_n(self, n: int) -> List[alphazero.StarGambitVariant]:
        """Sample n variants based on current distribution."""
        indices = np.random.choice(3, size=n, p=self.current_probs)
        return [self.VARIANT_MAP[self.VARIANT_NAMES[i]] for i in indices]

    def get_probs(self) -> Dict[str, float]:
        """Get current probability distribution as a dict."""
        return {v: float(self.current_probs[i]) for i, v in enumerate(self.VARIANT_NAMES)}

    def get_probs_at(self, iteration: int) -> Dict[str, float]:
        """Get probability distribution at a specific iteration."""
        probs = self._compute_probs(iteration)
        return {v: float(probs[i]) for i, v in enumerate(self.VARIANT_NAMES)}

    def get_weights(self) -> Dict[str, float]:
        """Get current distribution as weights for gating/evaluation."""
        return self.get_probs()

    def summary(self) -> str:
        """Return a string summary of current distribution."""
        probs = self.get_probs()
        return f"iter={self.current_iteration}: " + ", ".join(
            f"{v}={probs[v]*100:.1f}%" for v in self.VARIANT_NAMES
        )

    def __repr__(self):
        return f"SizeDistribution(iter={self.current_iteration}, probs={self.current_probs})"


def new_unified_game(variant: alphazero.StarGambitVariant) -> alphazero.StarGambitUnifiedGS:
    """Factory function to create a new unified game with the given variant."""
    return alphazero.StarGambitUnifiedGS(variant)


def new_game_multi(distribution: SizeDistribution) -> alphazero.StarGambitUnifiedGS:
    """Factory function that samples a variant and returns a unified game."""
    variant = distribution.sample()
    return alphazero.StarGambitUnifiedGS(variant)


if __name__ == "__main__":
    # Test the size distribution
    print("Testing SizeDistribution with default curriculum...")

    dist = SizeDistribution()

    # Show distribution at various iterations
    test_iters = [0, 50, 100, 150, 200, 250]
    for it in test_iters:
        dist.step(it)
        print(f"  {dist.summary()}")

    # Sample some variants
    print("\nSampling 100 variants at iteration 0:")
    dist.step(0)
    samples = dist.sample_n(100)
    counts = {}
    for s in samples:
        name = str(s).split('.')[-1].lower()
        counts[name] = counts.get(name, 0) + 1
    print(f"  Counts: {counts}")

    # Sample at iteration 200 (should be ~equal)
    print("\nSampling 100 variants at iteration 200:")
    dist.step(200)
    samples = dist.sample_n(100)
    counts = {}
    for s in samples:
        name = str(s).split('.')[-1].lower()
        counts[name] = counts.get(name, 0) + 1
    print(f"  Counts: {counts}")

    # Test game creation
    print("\nCreating games with new_game_multi:")
    dist.step(100)
    for i in range(5):
        game = new_game_multi(dist)
        print(f"  Game {i+1}: {game.get_variant_name()}")

    print("\nAll tests passed!")
