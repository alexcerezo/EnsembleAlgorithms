import os

# Global flag for showing plots. Set environment variable SHOW_PLOTS=1/true/yes to enable.
_show_plots = os.getenv("SHOW_PLOTS", "1")
SHOW_PLOTS = _show_plots.lower() in ("1", "true", "yes", "y")


def is_show_plots() -> bool:
    return SHOW_PLOTS
