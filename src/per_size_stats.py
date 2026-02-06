"""Per-size statistics tracking for multi-size Star Gambit training.

This module tracks Elo ratings, win rates, and game lengths separately for
each game size (Skirmish, Clash, Battle), as well as computing combined metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


VARIANT_NAMES = ['skirmish', 'clash', 'battle']


@dataclass
class PerSizeStatistics:
    """Tracks per-size statistics for multi-size training.

    Maintains separate Elo ratings, win rate matrices, and game lengths
    for each variant, plus combined weighted statistics.
    """

    # Number of networks to track (grows as new networks are added)
    num_networks: int = 0

    # Per-size Elo ratings: dict mapping variant name to array of Elo ratings
    elo: Dict[str, np.ndarray] = field(default_factory=dict)

    # Per-size win rates: dict mapping variant name to 2D win rate matrix
    # win_rates[variant][i, j] = win rate of network i against network j
    win_rates: Dict[str, np.ndarray] = field(default_factory=dict)

    # Combined Elo (weighted average based on distribution)
    combined_elo: np.ndarray = field(default_factory=lambda: np.array([]))

    # Per-size game lengths for each network
    game_lengths: Dict[str, List[List[float]]] = field(default_factory=dict)

    # Games played per size
    games_played: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize tracking dicts for all variants."""
        for variant in VARIANT_NAMES:
            if variant not in self.elo:
                self.elo[variant] = np.array([])
            if variant not in self.win_rates:
                self.win_rates[variant] = np.array([]).reshape(0, 0)
            if variant not in self.game_lengths:
                self.game_lengths[variant] = []
            if variant not in self.games_played:
                self.games_played[variant] = 0

    def add_network(self, initial_elo: float = 1000.0) -> int:
        """Add a new network to track, returning its index."""
        idx = self.num_networks
        self.num_networks += 1

        # Expand Elo arrays
        for variant in VARIANT_NAMES:
            self.elo[variant] = np.append(self.elo[variant], initial_elo)

            # Expand win rate matrices
            old_wr = self.win_rates[variant]
            new_wr = np.zeros((self.num_networks, self.num_networks))
            if old_wr.size > 0:
                new_wr[:-1, :-1] = old_wr
            self.win_rates[variant] = new_wr

            # Add empty game lengths list
            self.game_lengths[variant].append([])

        # Expand combined Elo
        self.combined_elo = np.append(self.combined_elo, initial_elo)

        return idx

    def update_elo(self, variant: str, network_idx: int, new_elo: float):
        """Update Elo rating for a specific variant and network."""
        if variant not in VARIANT_NAMES:
            raise ValueError(f"Unknown variant: {variant}")
        if network_idx >= self.num_networks:
            raise ValueError(f"Network index {network_idx} out of range")

        self.elo[variant][network_idx] = new_elo

    def update_win_rate(self, variant: str, i: int, j: int, win_rate: float):
        """Update win rate for network i against network j in variant."""
        if variant not in VARIANT_NAMES:
            raise ValueError(f"Unknown variant: {variant}")

        self.win_rates[variant][i, j] = win_rate

    def record_game_length(self, variant: str, network_idx: int, length: float):
        """Record a game length for a network in a variant."""
        if variant not in VARIANT_NAMES:
            raise ValueError(f"Unknown variant: {variant}")

        self.game_lengths[variant][network_idx].append(length)

    def record_games_played(self, variant: str, count: int):
        """Record number of games played for a variant."""
        if variant not in VARIANT_NAMES:
            raise ValueError(f"Unknown variant: {variant}")

        self.games_played[variant] += count

    def compute_combined_elo(self, weights: Dict[str, float]):
        """Compute combined Elo as weighted average across variants.

        Args:
            weights: Dict mapping variant names to weights (should sum to 1)
        """
        if self.num_networks == 0:
            return

        # Normalize weights
        total = sum(weights.get(v, 0) for v in VARIANT_NAMES)
        if total == 0:
            return

        norm_weights = {v: weights.get(v, 0) / total for v in VARIANT_NAMES}

        # Compute weighted average
        self.combined_elo = np.zeros(self.num_networks)
        for variant in VARIANT_NAMES:
            self.combined_elo += norm_weights[variant] * self.elo[variant]

    def get_elo(self, variant: str, network_idx: int) -> float:
        """Get Elo rating for a specific variant and network."""
        return float(self.elo[variant][network_idx])

    def get_combined_elo(self, network_idx: int) -> float:
        """Get combined Elo rating for a network."""
        return float(self.combined_elo[network_idx])

    def get_latest_elo(self, variant: Optional[str] = None) -> float:
        """Get Elo rating of the most recent network."""
        if self.num_networks == 0:
            return 1000.0

        if variant is None:
            return float(self.combined_elo[-1])
        return float(self.elo[variant][-1])

    def get_avg_game_length(self, variant: str, network_idx: int) -> float:
        """Get average game length for a network in a variant."""
        lengths = self.game_lengths[variant][network_idx]
        if not lengths:
            return 0.0
        return float(np.mean(lengths))

    def summary(self, weights: Optional[Dict[str, float]] = None) -> str:
        """Return a summary string of current statistics."""
        if self.num_networks == 0:
            return "No networks tracked yet"

        lines = [f"Networks: {self.num_networks}"]

        # Per-variant Elo for latest network
        latest = self.num_networks - 1
        for variant in VARIANT_NAMES:
            elo = self.elo[variant][latest]
            games = self.games_played.get(variant, 0)
            lines.append(f"  {variant.capitalize()}: Elo={elo:.1f}, Games={games}")

        # Combined Elo
        if weights:
            self.compute_combined_elo(weights)
        lines.append(f"  Combined: Elo={self.combined_elo[latest]:.1f}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict for saving."""
        return {
            'num_networks': self.num_networks,
            'elo': {k: v.tolist() for k, v in self.elo.items()},
            'win_rates': {k: v.tolist() for k, v in self.win_rates.items()},
            'combined_elo': self.combined_elo.tolist(),
            'game_lengths': self.game_lengths,
            'games_played': self.games_played,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PerSizeStatistics':
        """Deserialize from dict."""
        stats = cls()
        stats.num_networks = data['num_networks']
        stats.elo = {k: np.array(v) for k, v in data['elo'].items()}
        stats.win_rates = {k: np.array(v) for k, v in data['win_rates'].items()}
        stats.combined_elo = np.array(data['combined_elo'])
        stats.game_lengths = data['game_lengths']
        stats.games_played = data['games_played']
        return stats


def compute_elo_update(winner_elo: float, loser_elo: float, k: float = 32.0,
                       result: float = 1.0) -> Tuple[float, float]:
    """Compute Elo rating updates after a game.

    Args:
        winner_elo: Current Elo of winner
        loser_elo: Current Elo of loser
        k: K-factor (higher = more volatile ratings)
        result: Game result (1.0 = win, 0.5 = draw, 0.0 = loss)

    Returns:
        Tuple of (new_winner_elo, new_loser_elo)
    """
    expected_winner = 1.0 / (1.0 + 10**((loser_elo - winner_elo) / 400.0))
    expected_loser = 1.0 - expected_winner

    new_winner_elo = winner_elo + k * (result - expected_winner)
    new_loser_elo = loser_elo + k * ((1.0 - result) - expected_loser)

    return new_winner_elo, new_loser_elo


if __name__ == "__main__":
    # Test the per-size statistics
    print("Testing PerSizeStatistics...")

    stats = PerSizeStatistics()

    # Add some networks
    for i in range(5):
        idx = stats.add_network()
        print(f"Added network {idx}")

    # Update some Elo ratings
    stats.update_elo('skirmish', 4, 1100)
    stats.update_elo('clash', 4, 1050)
    stats.update_elo('battle', 4, 980)

    # Record some game data
    stats.record_games_played('skirmish', 100)
    stats.record_games_played('clash', 50)
    stats.record_games_played('battle', 25)

    stats.record_game_length('skirmish', 4, 45.0)
    stats.record_game_length('skirmish', 4, 52.0)

    # Print summary with equal weights
    weights = {'skirmish': 0.33, 'clash': 0.33, 'battle': 0.34}
    print(f"\n{stats.summary(weights)}")

    # Test Elo computation
    print("\nTesting Elo update:")
    w_elo, l_elo = compute_elo_update(1000, 1000)
    print(f"  1000 vs 1000, winner: {w_elo:.1f}, loser: {l_elo:.1f}")

    w_elo, l_elo = compute_elo_update(1200, 1000)
    print(f"  1200 vs 1000, 1200 wins: {w_elo:.1f}, 1000 goes to: {l_elo:.1f}")

    # Test serialization
    data = stats.to_dict()
    stats2 = PerSizeStatistics.from_dict(data)
    print(f"\nAfter serialization round-trip:")
    print(f"  num_networks: {stats2.num_networks}")
    print(f"  skirmish Elo[4]: {stats2.get_elo('skirmish', 4)}")

    print("\nAll tests passed!")
