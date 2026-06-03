"""Interactive UI for the tafl games (brandubh, open_tafl, tawlbwrdd).

Tafl moves are orthogonal slides: a move action encodes (from-square, direction,
destination) per ``tafl_helper.h::policyLocation``:

    new_loc     = action % (W + H)
    height_move = new_loc >= W           # slide along a column (changes row)
    piece_loc   = action // (W + H)      # from-square, row-major (h*W + w)

Coordinates are algebraic: file = column ``w`` (``a, b, …``), rank = ``H - h`` so
``a1`` is the bottom-left square, matching how the printed board reads.
"""

import re

import numpy as np

from game_ui import GameUI


class TaflUI(GameUI):
    def __init__(self, height: int, width: int):
        self.H = height
        self.W = width
        self.PW = width + height

    # -- coordinate helpers --------------------------------------------------

    def _sq_to_alg(self, h, w):
        return f"{chr(ord('a') + w)}{self.H - h}"

    def _alg_to_hw(self, file_ch, rank_str):
        w = ord(file_ch) - ord('a')
        try:
            rank = int(rank_str)
        except ValueError:
            return None
        h = self.H - rank
        if 0 <= w < self.W and 0 <= h < self.H:
            return h, w
        return None

    # -- action <-> (from, to) ----------------------------------------------

    def _decode(self, action):
        """action -> ((from_h, from_w), (to_h, to_w))."""
        new_loc = action % self.PW
        height_move = new_loc >= self.W
        piece_loc = action // self.PW
        pw = piece_loc % self.W
        ph = piece_loc // self.W
        if height_move:
            nh, nw = new_loc - self.W, pw
        else:
            nh, nw = ph, new_loc
        return (ph, pw), (nh, nw)

    def _encode(self, fr, to):
        """((from_h,from_w),(to_h,to_w)) -> action, or None if not an orthogonal slide."""
        (fh, fw), (th, tw) = fr, to
        if fw == tw and fh != th:
            offset = self.W + th          # column slide
        elif fh == th and fw != tw:
            offset = tw                   # row slide
        else:
            return None
        return (fh * self.W + fw) * self.PW + offset

    def _piece_glyph(self, canon, h, w):
        """King/Defender/Attacker glyph at (h, w) from canonical channels 0/1/2."""
        if canon[0, h, w] > 0.5:
            return "@"   # king
        if canon[1, h, w] > 0.5:
            return "O"   # defender
        if canon[2, h, w] > 0.5:
            return "X"   # attacker
        return ""

    # -- GameUI interface ----------------------------------------------------

    def display_board(self, gs) -> str:
        return str(gs)

    def format_move(self, gs, action) -> str:
        fr, to = self._decode(action)
        glyph = ""
        try:
            canon = np.asarray(gs.canonicalized())
            glyph = self._piece_glyph(canon, *fr)
        except Exception:
            pass
        prefix = f"{glyph} " if glyph else ""
        return f"{prefix}{self._sq_to_alg(*fr)}-{self._sq_to_alg(*to)}"

    def format_move_short(self, gs, action) -> str:
        fr, to = self._decode(action)
        return f"{self._sq_to_alg(*fr)}{self._sq_to_alg(*to)}"

    def parse_move(self, gs, input_str, valid_moves) -> int | None:
        s = input_str.strip().lower().replace("-", "").replace(" ", "")
        # Strip a leading piece glyph if present.
        if s and s[0] in "@ox":
            s = s[1:]
        squares = re.findall(r"([a-z])(\d+)", s)
        if len(squares) == 2:
            fr = self._alg_to_hw(*squares[0])
            to = self._alg_to_hw(*squares[1])
            if fr is None or to is None:
                return None
            action = self._encode(fr, to)
            if action is not None and 0 <= action < len(valid_moves) \
                    and valid_moves[action]:
                return action
            return None
        # Integer fallback.
        return super().parse_move(gs, input_str, valid_moves)

    def get_valid_move_descriptions(self, gs, valid_moves) -> list[tuple[int, str]]:
        return [(i, self.format_move_short(gs, i))
                for i in range(len(valid_moves)) if valid_moves[i]]

    def show_help(self, gs):
        last_file = chr(ord('a') + self.W - 1)
        print(f"Enter moves as algebraic squares, e.g. d4d7 or d4-d7 "
              f"(files a-{last_file}, ranks 1-{self.H}). Moves are orthogonal "
              f"slides. You can also enter an action ID.")

    def build_action_menu(self, gs, probs, valids, wld=None) -> list[tuple]:
        """Grouped menu: valid moves grouped by origin square, origins ordered
        by total policy mass."""
        valids = np.asarray(valids)
        valid_indices = np.where(valids > 0)[0]
        entries = []
        if wld is not None:
            entries.append(("info", f"Win: {wld[0]*100:.1f}%  "
                                    f"Loss: {wld[1]*100:.1f}%  "
                                    f"Draw: {wld[2]*100:.1f}%"))
        if len(valid_indices) == 0:
            return entries

        try:
            canon = np.asarray(gs.canonicalized())
        except Exception:
            canon = None

        # Group actions by origin square.
        by_origin = {}
        for a in valid_indices:
            fr, to = self._decode(int(a))
            by_origin.setdefault(fr, []).append((int(a), to))

        def origin_mass(fr):
            if probs is None:
                return 0.0
            return sum(probs[a] for a, _ in by_origin[fr])

        for fr in sorted(by_origin, key=origin_mass, reverse=True):
            glyph = self._piece_glyph(canon, *fr) if canon is not None else ""
            label = f"From {self._sq_to_alg(*fr)}"
            if glyph:
                label = f"{glyph} {label}"
            entries.append(("header", f"{label}:"))
            dests = by_origin[fr]
            if probs is not None:
                dests = sorted(dests, key=lambda d: probs[d[0]], reverse=True)
            for a, to in dests:
                text = self._sq_to_alg(*to)
                if probs is not None:
                    text += f"  [{probs[a]*100:5.1f}%]"
                entries.append(("action", a, text))
        return entries
