"""Star Gambit fire-impact heatmap trace matches the C++ ground truth.

The unified action heatmaps deposit fire-slot policy mass at the hex a shot
actually hits (traced along the cannon's line of fire), damage-weighted. This
validates game_runner._sg_fire_target_grid against C++ get_fire_info on the
unified 13x13 canvas: each fire action must deposit `damage` at exactly the
hex (and enemy type) the engine reports as the target.
"""
import numpy as np

import alphazero as az


def test_sg_fire_target_grid_matches_get_fire_info():
    from game_runner import _sg_fire_target_grid

    cls = az.StarGambitUnifiedClashGS
    rng = np.random.default_rng(5)
    checked = ok = 0
    for _ in range(120):
        g = cls(); g.randomize_start()
        for _ in range(70):
            legal = np.flatnonzero(np.array(g.valid_moves()))
            if len(legal) == 0:
                break
            c = np.array(g.canonicalized())                      # (36,13,13)
            valid = (c[0] > 0.5)[None]
            occ = (c[1:9] > 0.5).any(axis=0)[None]
            enemy = (c[5:9] > 0.5).any(axis=0)[None]
            face_oh = c[9:15]
            face = np.where(face_oh.max(axis=0) > 0.5, face_oh.argmax(axis=0), -1)[None]
            pres = [(c[ch] > 0.5)[None] for ch in (1, 2, 3)]
            for a in legal:
                slot, pos = a % 10, a // 10
                if slot < 5 or pos >= 169:
                    continue
                fi = g.get_fire_info(int(a))
                if not fi.has_target:
                    continue
                row, col = pos // 13, pos % 13
                if c[1:4, row, col].max() <= 0.5:
                    continue
                u = int(np.argmax(c[1:4, row, col]))
                sp = np.zeros((1, 13, 13, 10)); sp[0, row, col, slot] = 1.0
                fire_slots = np.zeros(10, dtype=bool); fire_slots[slot] = True
                grid = _sg_fire_target_grid(u, sp, pres[u], face, occ, enemy, valid, fire_slots)
                checked += 1
                nz = np.argwhere(grid > 0)
                assert len(nz) == 1, f"expected one impact hex, got {len(nz)}"
                tr, tc = nz[0]
                etype = int(np.argmax(c[5:9, tr, tc])) if c[5:9, tr, tc].max() > 0.5 else -1
                assert etype == fi.target_type, "impact hex is not the target's type"
                assert abs(grid[tr, tc] - fi.damage) < 1e-9, "wrong damage weight"
                ok += 1
            g.play_move(int(rng.choice(legal)))
            if g.scores() is not None:
                break
            if checked > 600:
                break
        if checked > 600:
            break
    assert checked > 100, f"too few fire actions exercised ({checked})"
    assert ok == checked
