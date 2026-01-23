"""
Plot Styling Helpers
====================

Small helper to keep plot styling consistent across pipeline scripts.
"""

from matplotlib import pyplot as plt


def set_plot_style(use_constrained_layout: bool = False) -> None:
    plt.style.use("classic")
    plt.rcParams["figure.constrained_layout.use"] = bool(use_constrained_layout)
