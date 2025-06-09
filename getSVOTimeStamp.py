import json

import pyzed.sl as sl
import argparse
from pathlib import Path
from loguru import logger
import datetime
import json


def get_from_json(json_directory: Path):
    json_file = json_directory / Path('timestamps.json')
    try:
        with open(json_file,'rt') as j:
            data = json.load(j)
    except FileNotFoundError:
        logger.error(f'Can not load {json_file}')
        return None
    timestamp = float(data['000000.png'])
    string =  datetime.datetime.fromtimestamp(timestamp/1000).strftime('%d_%m_%Y__%H_%M_%S')
    return timestamp, string

class GetSVOTimeStamp:
    def __init__(self,filename: Path):
        input_type = sl.InputType()
        input_type.set_from_svo_file(str(filename))  # Set init parameter to run from the .svo
        init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
        init.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
        # init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        self.cam = sl.Camera()

        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera opened successfully
            logger.error(f"Camera Open {status}.")
            raise Exception(f"Camera File Not Open: {status}")
        self.runtime = sl.RuntimeParameters()
        self.left_image = sl.Mat()

    def timestamp(self):
        err = self.cam.grab(self.runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.error('can not grab')
            return None
        timestamp = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        return timestamp, datetime.datetime.fromtimestamp(timestamp/1000).strftime('%d_%m_%Y__%H_%M_%S')


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=Path, help='Path to the SVO file',required=True)
opt = parser.parse_args()

p1 = Path("/home/d_phil/intentMAPS/ZED/Real_data/day1/batch1/YELLOWF3P1F.svo2")
p2 = Path("/home/d_phil/intentMAPS/ZED/Real_data/1/YELLOWF3P1F.svo2")

d1 = Path("/home/d_phil/intentMAPS/processed/real_proc1/1_with_pc/YELLOWF3P1F")

gts = GetSVOTimeStamp(p1)
timestamp,date = gts.timestamp()
print(timestamp)
print(date)


gts = GetSVOTimeStamp(p2)
timestamp,date = gts.timestamp()
print(timestamp)
print(date)


timestamp,date = get_from_json(d1)
print(timestamp)
print(date)

