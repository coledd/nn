# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Writes the title banner at the top of each example's README: series line,
# title, a one-sentence standfirst, three key figures and a line of tags,
# in the same layout for every example. Plain SVG with live text in the
# viewer's UI sans, so it stays sharp at any width and costs a few KB.

from xml.sax.saxutils import escape

from style import BLUE, GRID, INK, INK_2, MUTED, SURFACE, SVG_FONT

WIDTH, HEIGHT = 1600, 400


def _text(x, y, s, size, color, weight=400, anchor="start", spacing=0):
    return (f'<text x="{x}" y="{y}" font-family="{SVG_FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}" letter-spacing="{spacing}">{escape(s)}</text>')


def banner(path, number, eyebrow, title, standfirst, stats, tags):
    """`number` is the example's place in the series (None for the series
    itself); `standfirst` is a list of lines (pre-wrapped, ~60 characters each);
    `stats` three (value, label line 1, label line 2) tuples; `tags` a list
    of short strings."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}">',
           f'  <title>{escape(title)}</title>',
           f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{SURFACE}"/>',
           f'  <rect x="0" y="0" width="{WIDTH}" height="8" fill="{BLUE}"/>']
    series = eyebrow.upper() if number is None else f"EXAMPLE {number:02d}  ·  {eyebrow.upper()}"
    out.append('  ' + _text(72, 84, series, 17, BLUE, 600, spacing=2.2))
    out.append('  ' + _text(72, 152, title, 58, INK, 700, spacing=-0.5))
    for i, line in enumerate(standfirst):
        out.append('  ' + _text(72, 206 + i * 34, line, 24, INK_2))
    # Key figures: a thin rule, the value, a two-line label.
    x0, col = 1010, 196
    for i, (value, label1, label2) in enumerate(stats):
        x = x0 + i * col
        out.append(f'  <rect x="{x}" y="112" width="{col - 28}" height="3" fill="{BLUE}"/>')
        out.append('  ' + _text(x, 176, value, 44, INK, 650, spacing=-0.5))
        out.append('  ' + _text(x, 208, label1, 16.5, INK_2))
        out.append('  ' + _text(x, 230, label2, 16.5, INK_2))
    out.append(f'  <rect x="72" y="320" width="{WIDTH - 144}" height="1" fill="{GRID}"/>')
    out.append('  ' + _text(72, 358, "   ·   ".join(tags), 17, MUTED, 500, spacing=0.3))
    out.append('</svg>\n')
    with open(path, "w") as f:
        f.write("\n".join(out))
