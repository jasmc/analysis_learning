import shutil
from pathlib import Path

from tqdm import tqdm

path_orig_root = Path(r'E: ')

list_dir = [r'2024 09_Delay 2-P 4 planes JC neurons', r'2024 03_Delay 2-P 15 planes top part', r'2024 10_Delay 2-P 15 planes bottom part', '2024 10_Delay 2-P single plane']


for path_ in list_dir:

    path_orig = path_orig_root / path_

    path_end = Path(r'G:\2024 09_Delay 2-P multiple planes')
    path_end = path_end / path_orig.name

    # # For every folder in path_orig, copy the folders named 'Anatomical stack 1', 'Anatomical stack 2' if it exists and 'Imaging', plus the files 'Drift correction log.txt', 'Where is the plane.txt' to path_end
    # for folder in tqdm((path_orig / 'Imaging').iterdir()):
    #     if folder.is_dir():
    #         if (folder / 'Anatomical stack 1').exists():
    #             dest = path_end / folder.name / 'Anatomical stack 1'
    #             if not dest.exists():
    #                 shutil.copytree(folder / 'Anatomical stack 1', dest)
    #         if (folder / 'Anatomical stack 2').exists():
    #             dest = path_end / folder.name / 'Anatomical stack 2'
    #             if not dest.exists():
    #                 shutil.copytree(folder / 'Anatomical stack 2', dest)
    #         if (folder / 'Imaging').exists():
    #             dest = path_end / folder.name / 'Imaging'
    #             if not dest.exists():
    #                 shutil.copytree(folder / 'Imaging', dest)
    #         if (folder / 'Drift correction log.txt').exists():
    #             dest = path_end / folder.name / 'Drift correction log.txt'
    #             if not dest.exists():
    #                 shutil.copy2(folder / 'Drift correction log.txt', dest)
    #         if (folder / 'Where is the plane.txt').exists():
    #             dest = path_end / folder.name / 'Where is the plane.txt'
    #             if not dest.exists():
    #                 shutil.copy2(folder / 'Where is the plane.txt', dest)

    # For every file in path_orig\'Behavior', copy the to path_end
    for file in tqdm((path_orig / 'Behavior').iterdir()):
        if file.is_file() and file.suffix == '.txt':
            dest = path_end / file.name
            if not dest.exists():
                shutil.copy2(file, dest)
