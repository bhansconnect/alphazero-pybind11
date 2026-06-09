"""Canonicalization / symmetry consistency tests.

The core invariant under test: **if two states produce the same canonical
observation, they must expose the same canonical valid-move mask** (and hence
the same legal policy targets). The network only sees the canonical
observation, so any divergence means contradictory supervision.

- Star Gambit: a state S and its 180°-rotation+player-swap twin T are *defined*
  to canonicalize to the same observation (that is the player symmetry the
  canonicalization quotients by). They must therefore have identical canonical
  valids. They currently do NOT — the observation canonicalizes by pure 180°
  rotation while the policy/action space adds an L/R mirror via SLOT_MAP. The
  test is marked xfail(strict=True); it will flip to a failure (prompting marker
  removal) once the canonicalization is made consistent.

- Tafl: has no per-player spatial canonicalization (it uses a current-player
  plane), so it is structurally immune to the above. Its only spatial transform
  is the eightSym augmentation; we verify that augmentation transforms the board
  and the policy identically (a mirrored/rotated game stays a perfect image).
"""
import struct

import numpy as np
import pytest

import alphazero as az


# ---------------------------------------------------------------------------
# Star Gambit: rot180 + player-swap twin must have identical canonical valids
# ---------------------------------------------------------------------------

_SG_CLS = az.StarGambitClashGS  # Clash has fighters, cruisers, AND dreadnoughts


def _sg_parse(b):
    off = 0
    (num_units,) = struct.unpack_from("<I", b, off); off += 4
    units = []
    for _ in range(num_units):
        units.append(list(struct.unpack_from("<BBBBBbbBB", b, off))); off += 9
    res = list(struct.unpack_from("<8B", b, off)); off += 8
    cp = b[off]; off += 1
    (turn,) = struct.unpack_from("<I", b, off); off += 4
    hta, go = b[off], b[off + 1]; off += 2
    (winner,) = struct.unpack_from("<b", b, off); off += 1
    (hist_len,) = struct.unpack_from("<I", b, off); off += 4
    return dict(units=units, res=res, cp=cp, turn=turn, hta=hta, go=go, winner=winner)


def _sg_build(d):
    out = struct.pack("<I", len(d["units"]))
    for u in d["units"]:
        out += struct.pack("<BBBBBbbBB", *u)
    out += struct.pack("<8B", *d["res"])
    out += bytes([d["cp"]]) + struct.pack("<I", d["turn"])
    out += bytes([d["hta"], d["go"]]) + struct.pack("<b", d["winner"])
    out += struct.pack("<I", 0)  # zero repetition history (rep channel -> 0)
    return out


def _sg_make(raw):
    o = _SG_CLS.__new__(_SG_CLS)
    o.__setstate__(raw)
    return o


def _sg_norm(d):
    return {k: (v[:] if isinstance(v, list) else v) for k, v in d.items()}


def _sg_twin(d):
    """180° rotation (q,r)->(-q,-r), facing+3, swap players + reserves."""
    e = _sg_norm(d)
    for u in e["units"]:
        u[1] = 1 - u[1]              # player
        u[4] = (u[4] + 3) % 6        # facing
        u[5] = -u[5]                 # anchor_q
        u[6] = -u[6]                 # anchor_r
    e["res"] = e["res"][4:] + e["res"][:4]
    e["cp"] = 1 - d["cp"]
    return e


def _sg_collect_twin_pairs(n_pairs=15, max_games=400):
    rng = np.random.default_rng(7)
    pairs = []
    for _ in range(max_games):
        g = _SG_CLS(); g.randomize_start()
        for _ in range(70):
            legal = np.flatnonzero(np.array(g.valid_moves()))
            if len(legal) == 0:
                break
            has_fire = any((a % 10) >= 5 and (a // 10) < 121 for a in legal)
            if has_fire:
                d = _sg_parse(g.__getstate__())
                s = _sg_make(_sg_build(_sg_norm(d)))
                t = _sg_make(_sg_build(_sg_twin(d)))
                pairs.append((s, t))
                if len(pairs) >= n_pairs:
                    return pairs
            g.play_move(int(rng.choice(legal)))
            if g.scores() is not None:
                break
    return pairs


def test_sg_twin_has_identical_canonical_observation():
    """Sanity: the twin really is the same to the network (obs is consistent)."""
    pairs = _sg_collect_twin_pairs()
    assert pairs, "no fire-bearing states found"
    for s, t in pairs:
        assert np.array_equal(np.array(s.canonicalized()), np.array(t.canonicalized()))


def test_sg_same_canon_implies_same_valids():
    # Invariant: a state and its 180°-rotation+player-swap twin canonicalize to
    # the same observation, so they MUST expose the same canonical valid-move
    # mask. Historically broken: the observation canonicalized by pure 180°
    # rotation while the per-player action canon added an L/R mirror (SLOT_MAP),
    # so L/R-paired slots (rotate L/R, fire FL/FR/RL/RR) diverged.
    pairs = _sg_collect_twin_pairs()
    assert pairs, "no fire-bearing states found"
    for s, t in pairs:
        # precondition: identical canonical observation
        assert np.array_equal(np.array(s.canonicalized()), np.array(t.canonicalized()))
        # invariant: therefore identical canonical valid-move mask
        assert np.array_equal(np.array(s.valid_moves()), np.array(t.valid_moves()))


# ---------------------------------------------------------------------------
# Tafl: eightSym augmentation transforms board and policy identically
# ---------------------------------------------------------------------------

_W = _H = 11
_WH = _W + _H


def _t_decode(a):
    sq, rem = a // _WH, a % _WH
    fh, fw = sq // _W, sq % _W
    return ("row", fh, fw, rem) if rem < _W else ("col", fh, fw, rem - _W)


def _t_encode(kind, fh, fw, nl):
    return (fh * _W + fw) * _WH + (nl if kind == "row" else _W + nl)


def _t_mirror_w(a):
    kind, fh, fw, nl = _t_decode(a)
    fw2 = _W - 1 - fw
    return _t_encode("row", fh, fw2, _W - 1 - nl) if kind == "row" \
        else _t_encode("col", fh, fw2, nl)


def _t_rot90_cw(a):
    kind, fh, fw, nl = _t_decode(a)
    fh2, fw2 = fw, _H - 1 - fh
    return _t_encode("col", fh2, fw2, nl) if kind == "row" \
        else _t_encode("row", fh2, fw2, _H - 1 - nl)


def _t_board(g):
    return np.array(g.canonicalized())[:3]


@pytest.mark.parametrize("name,move_map,board_map", [
    ("mirror_w", _t_mirror_w, lambda bp: bp[:, :, ::-1]),
    ("rot90_cw", _t_rot90_cw, lambda bp: np.rot90(bp, k=-1, axes=(1, 2))),
])
def test_tafl_augmentation_board_policy_consistent(name, move_map, board_map):
    """A symmetry-mirrored game must stay a perfect image and never desync."""
    rng = np.random.default_rng(11)
    steps = 0
    for _ in range(60):
        ga, gb = az.OpenTaflGS(), az.OpenTaflGS()
        assert np.array_equal(board_map(_t_board(ga)), _t_board(gb)), f"{name}: start"
        for _ in range(50):
            va = np.flatnonzero(np.array(ga.valid_moves()))
            if len(va) == 0:
                break
            a = int(rng.choice(va))
            b = move_map(a)
            assert np.array(gb.valid_moves())[b], f"{name}: mirrored move illegal"
            ga.play_move(a); gb.play_move(b)
            steps += 1
            assert np.array_equal(board_map(_t_board(ga)), _t_board(gb)), f"{name}: desync"
            if ga.scores() is not None:
                break
    assert steps > 100
