"""Interactive arrow-key action selector for terminal-based play.

Provides a menu UI where users navigate with arrow keys and select with Enter.
Uses only stdlib (tty, termios, sys) - no external dependencies.
"""

import sys
import tty
import termios


def read_key():
    """Read a single keypress from stdin in raw mode.

    Returns:
        str: 'up', 'down', 'enter', 'tab', 'q', 'u', 'h', 'escape',
             or the character pressed.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

        if ch == '\x1b':  # Escape sequence
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                if ch3 == 'A':
                    return 'up'
                if ch3 == 'B':
                    return 'down'
                if ch3 == 'C':
                    return 'right'
                if ch3 == 'D':
                    return 'left'
                return 'escape'
            return 'escape'
        if ch == '\r' or ch == '\n':
            return 'enter'
        if ch == '\t':
            return 'tab'
        if ch == '\x03':  # Ctrl-C
            return 'ctrl-c'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ANSI escape helpers
REVERSE = '\033[7m'
DIM = '\033[2m'
BOLD = '\033[1m'
RESET_ATTR = '\033[0m'
CLEAR_LINE = '\033[2K'


class ActionSelector:
    """Interactive menu selector for game actions.

    Entry types (tuples):
        ("header", "Moves:")              - section header, not selectable
        ("action", action_id, "text...")  - selectable action item
        ("info", "Win: 85.3% ...")        - info line, not selectable

    Usage:
        selector = ActionSelector(entries, preselect=best_action_id)
        result = selector.run()
        # result: int (action_id), str (meta-command), or None (Tab pressed)
    """

    def __init__(self, entries, preselect=None):
        self.entries = entries
        self.selectable = [
            i for i, e in enumerate(entries) if e[0] == 'action'
        ]
        if not self.selectable:
            raise ValueError("No selectable entries in menu")

        # Find preselected index
        self.cursor = 0
        if preselect is not None:
            for si, idx in enumerate(self.selectable):
                if entries[idx][1] == preselect:
                    self.cursor = si
                    break

        self.total_lines = 0  # tracks lines printed for cursor-up

    def _format_entry(self, entry_idx):
        """Format a single entry for display."""
        entry = self.entries[entry_idx]
        kind = entry[0]

        if kind == 'header':
            return f"{BOLD}{entry[1]}{RESET_ATTR}"
        if kind == 'info':
            return f"  {DIM}{entry[1]}{RESET_ATTR}"
        if kind == 'action':
            is_selected = (
                entry_idx == self.selectable[self.cursor]
            )
            text = entry[2]
            if is_selected:
                return f"  {REVERSE}> {text}{RESET_ATTR}"
            else:
                return f"    {text}"
        return str(entry)

    def draw(self):
        """Print the full menu to stdout."""
        lines = []
        for i in range(len(self.entries)):
            lines.append(self._format_entry(i))

        lines.append('')
        lines.append(
            f"{DIM}[{RESET_ATTR}\u2191\u2193 Navigate{DIM}]  [{RESET_ATTR}"
            f"Enter Select{DIM}]  [{RESET_ATTR}"
            f"Tab Type{DIM}]  [{RESET_ATTR}"
            f"q Quit{DIM}]  [{RESET_ATTR}"
            f"u Undo{DIM}]{RESET_ATTR}"
        )

        output = '\n'.join(lines)
        sys.stdout.write(output + '\n')
        sys.stdout.flush()
        self.total_lines = len(lines)

    def redraw(self):
        """Move cursor up and redraw the menu in place."""
        # Move cursor up to the start of our menu
        if self.total_lines > 0:
            sys.stdout.write(f'\033[{self.total_lines}A')

        for i in range(len(self.entries)):
            sys.stdout.write(CLEAR_LINE + self._format_entry(i) + '\n')

        # Footer (blank line + hint line)
        sys.stdout.write(CLEAR_LINE + '\n')
        sys.stdout.write(
            CLEAR_LINE +
            f"{DIM}[{RESET_ATTR}\u2191\u2193 Navigate{DIM}]  [{RESET_ATTR}"
            f"Enter Select{DIM}]  [{RESET_ATTR}"
            f"Tab Type{DIM}]  [{RESET_ATTR}"
            f"q Quit{DIM}]  [{RESET_ATTR}"
            f"u Undo{DIM}]{RESET_ATTR}\n"
        )
        sys.stdout.flush()

    def cleanup(self):
        """Ensure cursor is at the bottom of the menu."""
        # Nothing to do - cursor is already at the bottom after draw/redraw
        pass

    def run(self):
        """Run the interactive selector.

        Returns:
            int: action_id if user selected an action
            str: 'quit', 'undo', 'help', 'status' for meta-commands
            None: user pressed Tab (switch to text input mode)
        """
        self.draw()

        try:
            while True:
                key = read_key()

                if key == 'up':
                    if self.cursor > 0:
                        self.cursor -= 1
                        self.redraw()
                elif key == 'down':
                    if self.cursor < len(self.selectable) - 1:
                        self.cursor += 1
                        self.redraw()
                elif key == 'enter':
                    entry = self.entries[self.selectable[self.cursor]]
                    self.cleanup()
                    return entry[1]  # action_id
                elif key == 'tab':
                    self.cleanup()
                    return None
                elif key == 'q':
                    self.cleanup()
                    return 'quit'
                elif key == 'u':
                    self.cleanup()
                    return 'undo'
                elif key == 'h':
                    self.cleanup()
                    return 'help'
                elif key == 's':
                    self.cleanup()
                    return 'status'
                elif key == 'v':
                    self.cleanup()
                    return 'valid'
                elif key == 'ctrl-c':
                    self.cleanup()
                    return 'quit'
                elif key == 'escape':
                    self.cleanup()
                    return None
        except (KeyboardInterrupt, EOFError):
            self.cleanup()
            return 'quit'
