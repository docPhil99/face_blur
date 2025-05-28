"""Sample code to reproject 3D points to 2D images"""
import cv2
import numpy as np
import json
from pathlib import Path
from loguru import logger
from utils import sort
from urllib.request import urlretrieve
import configparser

class GetConfig:
    def __init__(self, serial_number: str, config_path: Path = Path("./config")):
        self.serial_number = serial_number
        self.save_path = config_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.filename = self.save_path/Path(f"SN{self.serial_number}.conf")
        self.config = None
        self.url = f" http://calib.stereolabs.com/?SN={serial_number}"
        if not self._try_open():
            logger.info(f'Could not open file {self.filename}, download it from {self.url}' )
            raise FileNotFoundError(f'Could not open file {self.filename}')

    def _try_open(self):
        if self.filename.exists():
            self.config = configparser.ConfigParser()
            try:
                with open(self.filename, "rt") as cnf:
                    self.config.read_file(cnf)
            except FileNotFoundError:
                logger.error(f'Can not read config file {self.filename}')
                return False

            return True
        else:
            return False


class ReprojectPoints:
    def __init__(self,input_dir):
        self.input_dir  = input_dir
        config_path = input_dir/Path('config.json')
        try:
            with open(config_path) as f:
                self.config = json.load(f)
        except Exception as e:
            logger.exception(f'Failed to open {config_path}')
            raise e

        self.ZED_config = GetConfig(self.config['serial_number'], config_path=Path('/home/d_phil/intentMAPS/ZED/faceBlur/config'))
        body_path= input_dir/Path('bodies.json')
        try:
            with open(body_path) as f:
                self.bodies = json.load(f)
        except Exception as e:
            logger.exception(f'Failed to open {body_path}')
            raise e

        self.max_frame_number = max([int(x) for x in self.bodies.keys()])
        logger.debug(f'Max frame number: {self.max_frame_number}')
        self.frame_number = 0
        self.limg = None
        self.rimg = None

    def _drawLR_image(self):
        img = np.concatenate((self.limg, self.rimg), axis=1)
        sz = img.shape
        img = cv2.resize(img, (sz[1] // 2, sz[0] // 2))
        cv2.imshow('body', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            return False
        if key == ord('x') and self.frame_number < self.max_frame_number:
            self.frame_number = self.frame_number + 1
        if key == ord('z') and self.frame_number > 0:
            self.frame_number = self.frame_number - 1
        return True

    def _project3D_to_2D(self,camera_key:str):

        # cx = float(self.ZED_config.config[camera_key]["cx"])
        # cy = float(self.ZED_config.config[camera_key]["cy"])
        # fx = float(self.ZED_config.config[camera_key]["fx"])
        # fy = float(self.ZED_config.config[camera_key]["fy"])

        cx = float(self.config[camera_key]["cx"])
        cy = float(self.config[camera_key]["cy"])
        fx = float(self.config[camera_key]["fx"])
        fy = float(self.config[camera_key]["fy"])
        if "left" in camera_key.lower():
            #logger.debug('left camera')
            img = self.limg
        else:
            #logger.debug('right camera')
            img = self.rimg
        bods = self.bodies[str(self.frame_number)]["body_list"]
        for body in bods:
            kps_3d = body["keypoint"]  # all keypoints
            for kp in kps_3d:
                u = kp[0] / kp[2] * fx + cx
                v = kp[1] / kp[2] * fy + cy
                cv2.circle(img, (int(u), int(v)), 2, (255, 0, 0), 2)

    def start(self):
        while True:
            print(self.frame_number)
            path = input_dir / Path('left')/Path(f'{int(self.frame_number):06}.png')
            self.limg = cv2.imread(str(path))
            path = input_dir / Path('right') / Path(f'{int(self.frame_number):06}.png')
            self.rimg = cv2.imread(str(path))

            #self._project3D_to_2D("LEFT_CAM_2K")
            #self._project3D_to_2D("RIGHT_CAM_2K")
            self._project3D_to_2D("left_cam")
            self._project3D_to_2D("right_cam")

            # draw the image
            if not self._drawLR_image():
                break




if __name__=="__main__":
    input_dir =Path('/home/d_phil/intentMAPS/test_set/processed/BLACKB4P14S')
    logger.info(input_dir)
    rp = ReprojectPoints(input_dir)
    rp.start()
