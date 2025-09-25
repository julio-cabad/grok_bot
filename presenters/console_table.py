"""Console table rendering utilities for Spartan bot outputs."""
from typing import Iterable, Sequence


def render_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """Render an ASCII table to stdout.

    Args:
        headers: Ordered sequence of column headers.
        rows: Iterable of rows matching the header length.
    """
    rows_list = [tuple(str(cell) if cell is not None else "" for cell in row) for row in rows]

    if not rows_list:
        print("(sin datos)")
        return

    col_widths = [len(str(header)) for header in headers]

    for row in rows_list:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def build_separator(char: str = "+", horiz: str = "-") -> str:
        segments = [horiz * (width + 2) for width in col_widths]
        return char + char.join(segments) + char

    def format_row(row_values: Sequence[str]) -> str:
        cells = [f" {str(value).ljust(col_widths[idx])} " for idx, value in enumerate(row_values)]
        return "|" + "|".join(cells) + "|"

    separator = build_separator()
    header_line = format_row(headers)
    print(separator)
    print(header_line)
    print(separator.replace("-", "="))

    for row in rows_list:
        print(format_row(row))
        print(separator)
