"""Tiny helper so every analysis step can print AND save its results."""


class TextReport:
    """Collects lines of text, optionally echoing them to the console.

    Args:
        echo: If True (default) each logged line is also printed.

    Attributes:
        lines: The logged lines, in order.
    """

    def __init__(self, echo=True):
        self.echo = echo
        self.lines = []

    def log(self, *args):
        """Join ``args`` with spaces, store the line and optionally print it."""
        line = " ".join(str(a) for a in args)
        self.lines.append(line)
        if self.echo:
            print(line)

    def text(self):
        """Return everything logged so far as one string."""
        return "\n".join(self.lines)

    def save(self, path):
        """Write the logged text to ``path`` (UTF-8), creating parent folders."""
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text())
