"""Star Gambit game UI for the unified play agent.

Wraps the rich display and input parsing from star_gambit_play.py
into the GameUI interface.
"""

from game_ui import GameUI

# Re-use all the existing display/parsing logic from star_gambit_play
from star_gambit_play import (
    GameConfig,
    decode_action,
    format_action,
    print_unit_lists,
    print_actions_menu,
    parse_command,
    print_help as sg_print_help,
    GameContext,
)
import numpy as np


class StarGambitUI(GameUI):
    """Rich UI for Star Gambit with hex grid, unit selection, move naming."""

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg

    def display_board(self, gs) -> str:
        """Return board string and unit lists."""
        lines = [str(gs)]
        # Capture unit list output
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_unit_lists(gs)
        lines.append(f.getvalue())
        return "\n".join(lines)

    def parse_move(self, gs, input_str, valid_moves) -> int | None:
        """Parse Star Gambit move commands (m f1 f, f c1 l, d f ne, e, etc.)."""
        # Create a minimal context for parse_command
        import alphazero
        ctx = type('Ctx', (), {'game': gs, 'players': [None, None]})()

        result = parse_command(input_str, valid_moves, self.cfg, ctx)

        # parse_command returns special strings for non-move commands
        if isinstance(result, str):
            return None
        return result

    def format_move(self, gs, action) -> str:
        """Return human-friendly move description."""
        return format_action(action, self.cfg, gs)

    def get_valid_move_descriptions(self, gs, valid_moves) -> list[tuple[int, str]]:
        """Return (action_id, description) for all valid moves."""
        result = []
        for i in range(len(valid_moves)):
            if valid_moves[i]:
                desc = format_action(i, self.cfg, gs)
                result.append((i, desc))
        return result

    def show_help(self, gs):
        """Print Star Gambit help."""
        sg_print_help()
