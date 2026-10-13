# Documentation figures

The scripts here generate the figures in each example's README (`examples/<name>/figures/`): the title banner, charts of measured results, and figures drawn from each example's real or simulated data. The device illustrations and architecture diagrams are hand-written SVG and live only in the figures directories.

| File | Purpose |
|---|---|
| `style.py` | The shared look: palette, typeface, chart conventions. Every script imports it. |
| `banner.py` | The title banner at the top of each README. |
| `<example>.py` | The figures for one example. |

Requirements: Python 3 with `numpy` and `matplotlib`, plus the Noto Sans font. Scripts that plot an example's data need that data downloaded or built first. Run `make` in the example's directory, then:

```
python3 tools/docs/prosthetic_hand.py
```

Charts of results plot the numbers each README reports. If an example's results change, update the README and the matching values in its script together.
