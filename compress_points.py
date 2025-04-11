import gzip
import shutil
from loguru import logger
from pathlib import Path
import argparse

def compress_point_cloud(input_dir:Path,extension:str):
    output_directory = input_dir /Path('depth')
    logger.debug(f'Compressing point cloud in {output_directory}')

    for filename in output_directory.glob(f'*.{extension}'):
        logger.info(f'Compressing {filename}')
        with open(filename, 'rb') as f_in:
            with gzip.open(filename.with_suffix(filename.suffix+'.gz'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', '-i', type=Path, help='Path to the SVO file')
#parser.add_argument('--output_directory', '-o', type=Path, help='Path to the output directory', required=True)
parser.add_argument('--point_cloud_extension','-p',type=str,default='ply',help="Extension of point cloud files")
opt = parser.parse_args()
compress_point_cloud(opt.input_directory,opt.point_cloud_extension)