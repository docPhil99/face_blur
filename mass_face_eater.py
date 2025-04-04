from pathlib import Path
import argparse
import itertools
from loguru import logger
import subprocess
import os
"""
Wrapper around blur_face_svo.py to process a whole directory 
"""
logger.info('Mass face processor')
parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', '-i', type=Path, help='Path to the SVO file')
parser.add_argument('--output_directory', '-o', type=Path, help='Path to the output directory', required=True)
parser.add_argument('--no_blur', '-n', action='store_true', help="Don't blur the faces")
parser.add_argument('--no_depth', action='store_true', help="Don't store depth")
parser.add_argument('--show_3D', action='store_true', help="Display 3D bodies")
parser.add_argument('--no_display', action='store_true', help="Don't display images")
opt = parser.parse_args()

patterns = ["*.svo", "*.svo2"]
for input_file in itertools.chain.from_iterable(opt.input_directory.glob(pattern) for pattern in patterns):
    logger.info(f"Batch processing{input_file}")
    command_string = f'python blur_face_svo.py -i "{input_file}" -o "{opt.output_directory}"'
    if opt.no_blur:
        command_string += f" --no_blur"
    if opt.no_depth:
        command_string += f" --no_depth"
    if opt.show_3D:
        command_string += f" --show_3D"
    if opt.no_display:
        command_string += f" --no_display"
    logger.info(command_string)
    os.system(command_string)
