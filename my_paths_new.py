"""
Pipeline Path Configuration
===========================

Defines dataset selection and common path roots.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    path_home: Path
    path_results_save: Path
    fish_name: str


def build_config(path_home: Path, fish_name: str) -> PathConfig:
    return PathConfig(
        path_home=path_home,
        path_results_save=Path(r'F:\Results (paper)') / path_home.stem,
        fish_name=fish_name,
    )


# region Defaults
PATH_HOME = Path(r'D:\2024 09_Delay 2-P 4 planes JC neurons')
FISH_NAME = r'20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# endregion


CONFIG = build_config(PATH_HOME, FISH_NAME)

path_home = CONFIG.path_home
path_results_save = CONFIG.path_results_save
fish_name = CONFIG.fish_name
