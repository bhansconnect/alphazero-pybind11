"""Python tests for the StarGambitUnifiedGS binding and config integration."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import alphazero
from config import TrainConfig, GAME_REGISTRY, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UNIFIED_VARIANT_NAMES = ["skirmish", "showdown", "clash", "battle"]
NUM_VARIANTS = 4
UNIFIED_NUM_MOVES = 1709
UNIFIED_BOARD_DIM = 13
UNIFIED_CHANNELS = 36


def play_first_valid(game):
    """Play the first valid action. Returns False if already over."""
    if game.scores() is not None:
        return False
    valids = np.array(game.valid_moves())
    for a in range(len(valids)):
        if valids[a]:
            game.play_move(a)
            return True
    return False


# ---------------------------------------------------------------------------
# Static interface
# ---------------------------------------------------------------------------

class TestStaticInterface:
    def test_num_moves(self):
        assert alphazero.StarGambitUnifiedGS.NUM_MOVES() == UNIFIED_NUM_MOVES

    def test_canonical_shape(self):
        shape = alphazero.StarGambitUnifiedGS.CANONICAL_SHAPE()
        assert list(shape) == [UNIFIED_CHANNELS, UNIFIED_BOARD_DIM, UNIFIED_BOARD_DIM]

    def test_num_players(self):
        assert alphazero.StarGambitUnifiedGS.NUM_PLAYERS() == 2

    def test_num_symmetries(self):
        assert alphazero.StarGambitUnifiedGS.NUM_SYMMETRIES() == 2

    def test_registry_entries(self):
        expected = [
            "star_gambit_unified",
            "star_gambit_unified_skirmish",
            "star_gambit_unified_showdown",
            "star_gambit_unified_clash",
            "star_gambit_unified_battle",
        ]
        for name in expected:
            assert name in GAME_REGISTRY, f"{name!r} missing from GAME_REGISTRY"


# ---------------------------------------------------------------------------
# Canonical shape / game-type channels
# ---------------------------------------------------------------------------

class TestCanonicalObservation:
    def test_shape_all_variants(self):
        for v in range(NUM_VARIANTS):
            game = alphazero.StarGambitUnifiedGS(pinned_variant=v)
            obs = game.canonicalized()
            assert obs.shape == (UNIFIED_CHANNELS, UNIFIED_BOARD_DIM, UNIFIED_BOARD_DIM), \
                f"variant {v}: wrong shape {obs.shape}"

    def test_game_type_channels_one_hot(self):
        # Channel 32+v should be 1.0 at the board center (6,6), valid for all variants.
        for v in range(NUM_VARIANTS):
            game = alphazero.StarGambitUnifiedGS(pinned_variant=v)
            obs = game.canonicalized()
            for ch in range(32, 36):
                val = obs[ch, 6, 6]
                expected = 1.0 if ch == 32 + v else 0.0
                assert abs(val - expected) < 1e-5, \
                    f"variant {v} channel {ch}: expected {expected} got {val}"

    def test_game_type_channels_broadcast(self):
        # Game-type channel should have the same value at all valid hex positions.
        for v in range(NUM_VARIANTS):
            game = alphazero.StarGambitUnifiedGS(pinned_variant=v)
            obs = game.canonicalized()
            ch = 32 + v
            # All positions where valid_hex (channel 0) is 1.0 should have ch=1.0.
            valid_hex_mask = obs[0] > 0.5
            variant_ch = obs[ch]
            assert np.all(variant_ch[valid_hex_mask] > 0.9), \
                f"variant {v}: game-type channel not broadcast to all valid hexes"
            # Invalid positions should be 0.
            assert np.all(variant_ch[~valid_hex_mask] < 0.1), \
                f"variant {v}: game-type channel non-zero at invalid positions"

    def test_small_variant_outer_ring_zeros(self):
        # For Skirmish/Showdown/Clash (11x11), channel 0 outer ring should be all zeros.
        for v in range(3):
            game = alphazero.StarGambitUnifiedGS(pinned_variant=v)
            obs = game.canonicalized()
            ch0 = obs[0]
            assert np.all(ch0[0, :] == 0), f"variant {v}: row 0 non-zero"
            assert np.all(ch0[12, :] == 0), f"variant {v}: row 12 non-zero"
            assert np.all(ch0[:, 0] == 0), f"variant {v}: col 0 non-zero"
            assert np.all(ch0[:, 12] == 0), f"variant {v}: col 12 non-zero"

    def test_battle_outer_ring_has_valid_hexes(self):
        # Battle (13x13) should have valid hexes in outer rows.
        game = alphazero.StarGambitUnifiedGS(pinned_variant=3)
        obs = game.canonicalized()
        ch0 = obs[0]
        outer_max = max(ch0[0].max(), ch0[12].max(), ch0[:, 0].max(), ch0[:, 12].max())
        assert outer_max > 0.5, "Battle: no valid hexes in outer ring"


# ---------------------------------------------------------------------------
# Action remapping
# ---------------------------------------------------------------------------

class TestActionRemapping:
    def test_valid_moves_size(self):
        for v in range(NUM_VARIANTS):
            game = alphazero.StarGambitUnifiedGS(pinned_variant=v)
            valids = np.array(game.valid_moves())
            assert len(valids) == UNIFIED_NUM_MOVES, f"variant {v}: wrong valid_moves size"

    def test_small_variant_no_outer_ring_moves(self):
        # Skirmish valid spatial moves should not be in outer ring positions.
        SMALL_SPATIAL = 11 * 11 * 10
        UNIFIED_SPATIAL = UNIFIED_BOARD_DIM * UNIFIED_BOARD_DIM * 10
        for v in range(3):
            game = alphazero.StarGambitUnifiedGS(pinned_variant=v)
            valids = np.array(game.valid_moves())
            for action in range(UNIFIED_SPATIAL):
                if not valids[action]:
                    continue
                flat_pos = action // 10
                row = flat_pos // UNIFIED_BOARD_DIM
                col = flat_pos % UNIFIED_BOARD_DIM
                assert row > 0,  f"variant {v} action {action} in border row 0"
                assert row < 12, f"variant {v} action {action} in border row 12"
                assert col > 0,  f"variant {v} action {action} in border col 0"
                assert col < 12, f"variant {v} action {action} in border col 12"

    def test_deploy_at_unified_offset(self):
        UNIFIED_DEPLOY_OFFSET = 1690
        UNIFIED_END_TURN_OFFSET = 1708
        for v in range(NUM_VARIANTS):
            game = alphazero.StarGambitUnifiedGS(pinned_variant=v)
            valids = np.array(game.valid_moves())
            # Turn 1: should have deploy actions at unified offset, not at small offset.
            has_deploy = any(valids[UNIFIED_DEPLOY_OFFSET:UNIFIED_END_TURN_OFFSET])
            assert has_deploy, f"variant {v}: no deploy at unified offset on turn 1"
            if v < 3:
                SMALL_DEPLOY_OFFSET = 1210
                SMALL_END_TURN_OFFSET = 1228
                has_small_deploy = any(valids[SMALL_DEPLOY_OFFSET:SMALL_END_TURN_OFFSET])
                assert not has_small_deploy, f"variant {v}: deploy at small offset (wrong)"

    def test_can_play_moves(self):
        # Basic smoke: can play through several turns without crashing.
        for v in range(NUM_VARIANTS):
            game = alphazero.StarGambitUnifiedGS(pinned_variant=v)
            for _ in range(10):
                if game.scores() is not None:
                    break
                ok = play_first_valid(game)
                assert ok, f"variant {v}: no valid moves (game not over)"


# ---------------------------------------------------------------------------
# Variant selection
# ---------------------------------------------------------------------------

class TestVariantSelection:
    def test_pinned_variant(self):
        for v in range(NUM_VARIANTS):
            game = alphazero.StarGambitUnifiedGS(pinned_variant=v)
            for _ in range(5):
                game.randomize_start()
                assert game.get_variant_id() == v

    def test_pinned_subclasses(self):
        expected = {
            "StarGambitUnifiedSkirmishGS": 0,
            "StarGambitUnifiedShowdownGS": 1,
            "StarGambitUnifiedClashGS": 2,
            "StarGambitUnifiedBattleGS": 3,
        }
        for cls_name, v in expected.items():
            cls = getattr(alphazero, cls_name)
            game = cls()
            for _ in range(5):
                game.randomize_start()
                assert game.get_variant_id() == v, f"{cls_name}: expected variant {v}"

    def test_random_variant_distribution(self):
        # With equal probs, all 4 variants should appear over many samples.
        game = alphazero.StarGambitUnifiedGS(pinned_variant=-1, probs=[0.25, 0.25, 0.25, 0.25])
        counts = [0] * 4
        for _ in range(200):
            game.randomize_start()
            counts[game.get_variant_id()] += 1
        for v in range(4):
            assert counts[v] > 0, f"variant {v} never appeared with equal probs"

    def test_biased_variant_distribution(self):
        # With [1,0,0,0], should always get Skirmish.
        game = alphazero.StarGambitUnifiedGS(pinned_variant=-1, probs=[1.0, 0.0, 0.0, 0.0])
        for _ in range(20):
            game.randomize_start()
            assert game.get_variant_id() == 0


# ---------------------------------------------------------------------------
# Copy and equality
# ---------------------------------------------------------------------------

class TestCopy:
    def test_copy_same_state(self):
        game = alphazero.StarGambitUnifiedGS(pinned_variant=0)
        for _ in range(4):
            play_first_valid(game)
        copy = game.copy()
        assert copy.get_variant_id() == game.get_variant_id()
        assert copy.current_turn() == game.current_turn()
        assert copy.current_player() == game.current_player()
        assert np.allclose(copy.canonicalized(), game.canonicalized())

    def test_copy_independence(self):
        game = alphazero.StarGambitUnifiedGS(pinned_variant=0)
        copy = game.copy()
        play_first_valid(copy)
        assert copy.current_turn() != game.current_turn()


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_load_game_from_registry(self):
        config = TrainConfig(game="star_gambit_unified")
        config.validate()
        Game = config.Game
        assert Game is alphazero.StarGambitUnifiedGS

    def test_variant_fractions_valid(self):
        config = TrainConfig(
            game="star_gambit_unified",
            variant_fractions={"skirmish": 0.25, "showdown": 0.25, "clash": 0.25, "battle": 0.25},
        )
        config.validate()  # Should not raise.

    def test_variant_fractions_must_sum_to_one(self):
        config = TrainConfig(
            game="star_gambit_unified",
            variant_fractions={"skirmish": 0.3, "showdown": 0.3, "clash": 0.3, "battle": 0.3},
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            config.validate()

    def test_variant_fractions_unknown_key(self):
        config = TrainConfig(
            game="star_gambit_unified",
            variant_fractions={"skirmish": 0.5, "battle": 0.3, "unknown_variant": 0.2},
        )
        with pytest.raises(ValueError, match="Unknown variant"):
            config.validate()

    def test_variant_mixing_mode_invalid(self):
        config = TrainConfig(
            game="star_gambit_unified",
            variant_mixing_mode="invalid_mode",
        )
        with pytest.raises(ValueError, match="variant_mixing_mode"):
            config.validate()

    def test_pinned_variants_in_registry(self):
        for suffix in ("skirmish", "showdown", "clash", "battle"):
            name = f"star_gambit_unified_{suffix}"
            config = TrainConfig(game=name)
            config.validate()
            Game = config.Game
            game = Game()
            game.randomize_start()
            expected_id = ["skirmish", "showdown", "clash", "battle"].index(suffix)
            assert game.get_variant_id() == expected_id


# ---------------------------------------------------------------------------
# Compute unified probs helper (game_runner logic)
# ---------------------------------------------------------------------------

class TestComputeUnifiedProbs:
    """Test the _compute_unified_probs helper from game_runner."""

    def _get_helper(self):
        import game_runner
        return game_runner._compute_unified_probs

    def test_game_based_equal(self):
        compute = self._get_helper()
        config = TrainConfig(game="star_gambit_unified")
        probs = compute(config)
        assert len(probs) == 4
        assert all(abs(p - 0.25) < 1e-5 for p in probs)

    def test_game_based_custom_fractions(self):
        compute = self._get_helper()
        config = TrainConfig(
            game="star_gambit_unified",
            variant_fractions={"skirmish": 0.5, "showdown": 0.2, "clash": 0.2, "battle": 0.1},
        )
        probs = compute(config)
        assert abs(probs[0] - 0.5) < 1e-4
        assert abs(probs[3] - 0.1) < 1e-4

    def test_sample_based_adjusts(self):
        compute = self._get_helper()
        # Equal target, but Battle generated 2x as many samples as others.
        config = TrainConfig(
            game="star_gambit_unified",
            variant_fractions={"skirmish": 0.25, "showdown": 0.25, "clash": 0.25, "battle": 0.25},
            variant_mixing_mode="sample_based",
        )
        # actual fracs: [0.2, 0.2, 0.2, 0.4], targets: [0.25]*4
        # adjusted = [0.25/0.2, 0.25/0.2, 0.25/0.2, 0.25/0.4] = [1.25, 1.25, 1.25, 0.625]
        # normalized: small = 1.25/4.375 ≈ 0.2857, battle = 0.625/4.375 ≈ 0.1429
        sample_counts = [100, 100, 100, 200]
        probs = compute(config, prev_sample_counts=sample_counts)
        assert abs(sum(probs) - 1.0) < 1e-5
        expected_small = 1.25 / 4.375
        expected_battle = 0.625 / 4.375
        for vi in range(3):
            assert abs(probs[vi] - expected_small) < 1e-4, \
                f"variant {vi}: expected {expected_small:.4f} got {probs[vi]:.4f}"
        assert abs(probs[3] - expected_battle) < 1e-4, \
            f"battle: expected {expected_battle:.4f} got {probs[3]:.4f}"

    def test_sample_based_under_represented_boosted(self):
        compute = self._get_helper()
        # Skirmish generated only half the expected samples.
        config = TrainConfig(
            game="star_gambit_unified",
            variant_fractions={"skirmish": 0.25, "showdown": 0.25, "clash": 0.25, "battle": 0.25},
            variant_mixing_mode="sample_based",
        )
        # actual fracs: [0.1, 0.3, 0.3, 0.3], targets: [0.25]*4
        # adjusted = [0.25/0.1, 0.25/0.3, 0.25/0.3, 0.25/0.3] = [2.5, 0.833, 0.833, 0.833]
        sample_counts = [100, 300, 300, 300]
        probs = compute(config, prev_sample_counts=sample_counts)
        assert abs(sum(probs) - 1.0) < 1e-5
        assert probs[0] > probs[1], "Under-represented Skirmish should get higher game prob"
        expected_skirmish = 2.5 / (2.5 + 3 * (0.25 / 0.3))
        assert abs(probs[0] - expected_skirmish) < 1e-4

    def test_sample_based_zero_samples_handled(self):
        compute = self._get_helper()
        # One variant got 0 samples — fallback keeps it alive (target * 4).
        config = TrainConfig(
            game="star_gambit_unified",
            variant_fractions={"skirmish": 0.25, "showdown": 0.25, "clash": 0.25, "battle": 0.25},
            variant_mixing_mode="sample_based",
        )
        sample_counts = [0, 200, 200, 100]
        probs = compute(config, prev_sample_counts=sample_counts)
        assert abs(sum(probs) - 1.0) < 1e-5
        assert probs[0] > 0, "Zero-sample variant should still get nonzero prob"

    def test_sample_based_fallback_without_counts(self):
        compute = self._get_helper()
        config = TrainConfig(
            game="star_gambit_unified",
            variant_fractions={"skirmish": 0.4, "showdown": 0.3, "clash": 0.2, "battle": 0.1},
            variant_mixing_mode="sample_based",
        )
        # No previous counts → falls back to target fractions.
        probs = compute(config, prev_sample_counts=None)
        assert abs(probs[0] - 0.4) < 1e-4
        assert abs(probs[3] - 0.1) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
