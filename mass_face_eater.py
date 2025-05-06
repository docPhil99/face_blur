import sys
from pathlib import Path
import argparse
import itertools
from loguru import logger
import subprocess
import os



def compress_point_cloud(input_file:Path,output_directory:Path,extension:str):
    out_dir = input_file.stem
    output_directory = output_directory / out_dir /Path('depth')
    logger.debug(f'Compressing point cloud in {output_directory}')
    gzip_string = output_directory / Path(f'*.{extension}')
    logger.debug(gzip_string)
    cmd = "gzip "+str(gzip_string)+' &'
    os.system(cmd)
    logger.debug('done compressing point cloud')
    #qleobj(f_in, f_out)

logger.add("logs/mass_face_eater.log")

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
parser.add_argument('--no_point_cloud', action='store_true', help="Don't save point cloud")
parser.add_argument('--point_cloud_extension','-p',type=str,default='ply',help="Extension of point cloud files")
parser.add_argument('--dry_run', action='store_true', help="Dry run")
parser.add_argument("--compress_point_cloud", action='store_true', help="compress the point cloud file")
parser.add_argument('--skip_list', type=Path, help='Optional file containing files to skip, same format as logs/proc_list.txt')
parser.add_argument('--skip_by_dir',action='store_true',help="Skip files if the output directory exists with the same name")
opt = parser.parse_args()
skip_file_list = []
output_dir_list = []
if opt.skip_list:
    try:
        with open(opt.skip_list,'rt') as f:
            dat = f.read()
            skip_file_list = dat.split("\n")
            logger.info('Using skip list')
    except FileNotFoundError:
        logger.error(f"File not found: {opt.skip_list} ")
        sys.exit(-1)
if opt.skip_by_dir:
    output_dir_list = [f.name for f in opt.output_directory.iterdir() if f.is_dir()]
    logger.info(f'Found {output_dir_list}')

with open("logs/proc_list.txt",'wt') as f:
    logger.info(f'Processing {opt.input_directory}')
    patterns = ["**/*.svo", "**/*.svo2"]
    for input_file in itertools.chain.from_iterable(opt.input_directory.glob(pattern) for pattern in patterns):
        if str(input_file) in skip_file_list:
            logger.info(f'Skipping file: {input_file}')
            continue
        if str(input_file.stem) in output_dir_list:
            logger.info(f'Skipping file: {input_file}')
            continue
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
        if opt.no_point_cloud:
            command_string += f" --no_point_cloud"
        command_string += f" --point_cloud_extension {opt.point_cloud_extension}"
        logger.info(command_string)
        if not opt.dry_run:
            os.system(command_string)
        if opt.compress_point_cloud:
            compress_point_cloud(input_file, opt.output_directory,opt.point_cloud_extension)
        print(input_file,file=f,flush=True)