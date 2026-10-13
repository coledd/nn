# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# The banner at the top of examples/README.md.
#
#   python3 tools/docs/series.py

import os

from banner import banner

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "figures")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    banner(os.path.join(OUT, "banner.svg"), None, "Neural network library · examples",
           "Embedded Neural Networks",
           ["Eight complete applications, from handwritten digits",
            "to prosthetic hands, each built to run on a microcontroller."],
           [("8", "worked examples,", "data to firmware"),
            ("5", "layer types: FC, CNN,", "RNN, GRU, LSTM"),
            ("5", "trained on real", "public datasets")],
           ["Vision", "Audio", "Motion", "Vibration", "Biosignals", "STM32", "C, no dependencies"])
    print("wrote", os.path.join(OUT, "banner.svg"))
