"""
Pipeline Path Configuration
===========================

Defines dataset selection and common path roots used across the imaging pipeline.
Edit PATH_HOME and FISH_NAME below, or use build_config() at runtime.
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

FISH_NAME_OPTIONS = (
    '20240909_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf',
    '20240912_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20240415_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf',
    '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf',
    '20240912_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20240927_03_control_2p-6_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf',
    '20241013_01_control_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf',
    '20241013_02_control_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf',
    '20241024_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf',
    '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf',
    '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf',
    '20240926_03_trace_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf',
    '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf',
    '20241024_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20241007_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf',
    '20240930_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf',
    '20241001_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20241001_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20241014_03_trace_2p-6_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20241014_02_trace_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20241014_01_trace_2p-4_mitfaminusminus,elavl3h2bgcamp6f_6dpf',
    '20241013_03_control_2p-3_mitfaminusminus,elavl3h2bgcamp6f_5dpf',
    '20241010_03_trace_2p-3_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20241010_01_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20241009_03_delay_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf',
    '20241009_02_delay_2p-8_mitfaminusminus,elavl3h2bgcamp6f_5dpf',
    '20241009_01_delay_2p-7_mitfaminusminus,elavl3h2bgcamp6f_5dpf',
    '20241007_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf',
    '20241008_02_delay_2p-6_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
    '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf',
    '20241004_03_delay_2p-1_mitfaminusminus,ca8e1bgcamp6s_6dpf',
    '20241004_02_delay_2p-1_mitfaminusminus,ca8e1bgcamp6s_6dpf',
    '20241015_02_delay_2p-8_mitfaminusminus,ca8e1bgcamp6s_6dpf',
    '20241003_01_delay_2p-1_mitfaminusminus,ca8e1bgcamp6s_5dpf',
    '20241002_03_delay_2p-1_mitfaminusminus,ca8e1bgcamp6s_6dpf',
    '20241002_02_delay_2p-1_mitfaminusminus,ca8e1bgcamp6s_6dpf',
    '20241016_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf',
    '20240417_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_5dpf',
    '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf',
)
# endregion


CONFIG = build_config(PATH_HOME, FISH_NAME)

path_home = CONFIG.path_home
path_results_save = CONFIG.path_results_save
fish_name = CONFIG.fish_name


