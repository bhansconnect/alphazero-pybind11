"""Game UI base class and registry for pluggable play interfaces."""


class GameUI:
    """Base class for game-specific play UI."""

    def display_board(self, gs) -> str:
        """Return string representation of the board."""
        return str(gs)

    def parse_move(self, gs, input_str, valid_moves) -> int | None:
        """Parse human input into action ID. Return None if invalid."""
        try:
            action = int(input_str)
            if 0 <= action < len(valid_moves) and valid_moves[action]:
                return action
            return None
        except ValueError:
            return None

    def format_move(self, gs, action) -> str:
        """Return human-friendly name for an action."""
        return str(action)

    def get_valid_move_descriptions(self, gs, valid_moves) -> list[tuple[int, str]]:
        """Return list of (action_id, description) for valid moves."""
        return [(i, str(i)) for i in range(len(valid_moves)) if valid_moves[i]]

    def show_help(self, gs):
        """Print game-specific help text."""
        print("Enter action ID (integer). Type 'help' for valid moves.")


# Registry mapping game names to UI factory functions.
# Lazy imports to avoid loading star_gambit_play unless needed.
GAME_UI_REGISTRY: dict[str, callable] = {}


def _register_star_gambit_uis():
    """Register Star Gambit UIs lazily."""
    try:
        from star_gambit_ui import StarGambitUI
        from star_gambit_play import SKIRMISH, CLASH, BATTLE
        GAME_UI_REGISTRY["star_gambit_skirmish"] = lambda: StarGambitUI(SKIRMISH)
        GAME_UI_REGISTRY["star_gambit_clash"] = lambda: StarGambitUI(CLASH)
        GAME_UI_REGISTRY["star_gambit_battle"] = lambda: StarGambitUI(BATTLE)
    except ImportError:
        pass


def get_game_ui(game_name: str) -> GameUI:
    """Get the appropriate GameUI for a game name."""
    if not GAME_UI_REGISTRY:
        _register_star_gambit_uis()
    if game_name in GAME_UI_REGISTRY:
        return GAME_UI_REGISTRY[game_name]()
    return GameUI()
