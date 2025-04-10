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
opt = parser.parse_args()

with open("logs/proc_list.txt",'wt') as f:
    logger.info(f'Processing {opt.input_directory}')
    patterns = ["**/*.svo", "**/*.svo2"]
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
        if opt.no_point_cloud:
            command_string += f" --no_point_cloud"
        command_string += f" --point_cloud_extension {opt.point_cloud_extension}"
        logger.info(command_string)
        if not opt.dry_run:
            os.system(command_string)
        if opt.compress_point_cloud:
            compress_point_cloud(input_file, opt.output_directory,opt.point_cloud_extension)
        print(input_file,file=f,flush=True)