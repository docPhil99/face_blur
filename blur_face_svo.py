

"""
    Read SVO and convert to images and depth array.
"""
from retinaface import RetinaFace
import sys
import pyzed.sl as sl
import cv2
import argparse
import os
from loguru import logger
from pathlib import Path
import numpy as np
import gzip

class face_blur:
    def __init__(self):
        pass

    @staticmethod
    def blur(image):
        resp = RetinaFace.detect_faces(image)
        for key, face in resp.items():
            bbox = face['facial_area']
            image = face_blur._blur_face(image, bbox)
        return image

    @staticmethod
    def _blur_face(image, bbox):
        face = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        face_blur = cv2.blur(face, (200, 200))
        image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = face_blur
        return image

class SVO_Process:
    def __init__(self,opt):
        self.opt = opt
        self.filepath = opt.input
        self.cam = None
        self.svo_image =None
        self.runtime = None
        self.left_image_path = self.opt.output_directory/Path('left')
        self.right_image_path = self.opt.output_directory/Path('right')
        self.depth_image_path = self.opt.output_directory/Path('depth')
        self.left_image_path.mkdir(parents=True, exist_ok=True)
        self.right_image_path.mkdir(parents=True, exist_ok=True)
        self.depth_image_path.mkdir(parents=True, exist_ok=True)

        self._open_file()

    @staticmethod
    def progress_bar(percent_done, bar_length=50):
        # Display progress bar
        done_length = int(bar_length * percent_done / 100)
        bar = '=' * done_length + '-' * (bar_length - done_length)
        sys.stdout.write('[%s] %i%s\r' % (bar, percent_done, '%'))
        sys.stdout.flush()

    def _open_file(self):
        input_type = sl.InputType()
        input_type.set_from_svo_file(str(self.filepath))  # Set init parameter to run from the .svo
        init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
        init.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)
        #init.depth_mode = sl.DEPTH_MODE.PERFORMANCE # TODO
        self.cam = sl.Camera()

        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera opened succesfully
            logger.error(f"Camera Open {status}.")
            raise Exception(f"Camera File Not Open: {status}")

        # Set a maximum resolution, for visualisation confort
        resolution = self.cam.get_camera_information().camera_configuration.resolution
        config = self.cam.get_camera_information()
        #TODO save this information

        self.runtime = sl.RuntimeParameters()
        # Prepare single image containers
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        self.depth_image = sl.Mat()
        self.depth_map = sl.Mat()
        self.mat = sl.Mat()
        self.svo_frame_rate = self.cam.get_init_parameters().camera_fps
        self.nb_frames = self.cam.get_svo_number_of_frames()
        logger.info(f"SVO contains {self.nb_frames}  frames at {self.svo_frame_rate} fps")

    def _save_image(self,filename,image, blur=False):
        logger.debug(f"Saved image :  {filename}")
        img = image.get_data()
        #drop alpha
        img = img[:,:,0:3]
        if blur:
            img = face_blur.blur(img)
        cv2.imwrite(str(filename),img)
    def _save_images(self):

        #left
        filename = self.left_image_path/Path(f'{self.svo_position:06}.png')
        self._save_image(filename,self.left_image,blur=not self.opt.no_blur)

        filename = self.right_image_path / Path(f'{self.svo_position:06}.png')
        self._save_image(filename,self.right_image,blur=not self.opt.no_blur)
        # depth
        filename = self.depth_image_path / Path(f'{self.svo_position:06}.png')
        self._save_image(filename,self.depth_image,blur=False)

        try:
            with gzip.GzipFile(filename,"w") as f:
                np.save(file=f, arr=self.depth_map.get_data())
        except Exception as e:
            logger.exception(f'Could not save {filename}')

    def process_loop(self):
        self.run_flag = True
        while self.run_flag:
            err = self.cam.grab(self.runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self.svo_position = self.cam.get_svo_position()
                self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT)
                self.cam.retrieve_image(self.right_image, sl.VIEW.RIGHT)
                self.cam.retrieve_image(self.depth_image, sl.VIEW.DEPTH)
                self.cam.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                self._save_images()

                cv2.imshow("ViewL", self.left_image.get_data())  # dislay both images to cv2
                cv2.imshow("ViewR", self.right_image.get_data())  # dislay both images to cv2
                cv2.imshow("ViewD", self.depth_image.get_data())  # dislay both images to cv2

                key = cv2.waitKey(10)
                if key == ord('q'):
                    self.run_flag = False

                self.progress_bar(self.svo_position/ self.nb_frames * 100, 30)
            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:  # Check if the .svo has ended
                self.progress_bar(100, 30)
                logger.info("SVO end has been reached.")
                break
            else:
                logger.error("Grab ZED : ", err)
                break

    def shutdown(self):
        cv2.destroyAllWindows()
        self.cam.close()


def main(opt):
    proc  = SVO_Process(opt)
    proc.process_loop()
    proc.shutdown()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i', type=Path, help='Path to the SVO file', required=True)
    parser.add_argument('--output_directory','-o', type=Path, help='Path to the output directory', required=True)
    parser.add_argument('--no_blur','-n', action='store_true', help="Don't blur the faces")
    opt = parser.parse_args()
    if not opt.input.suffix==".svo" and not opt.input.suffix==".svo2":
        print("--input_svo_file parameter should be a .svo file but is not : ", opt.input, "Exit program.")
        exit()
    if not os.path.isfile(opt.input):
        print("--input_svo_file parameter should be an existing file but is not : ", opt.input,
              "Exit program.")
        exit()
    out_dir = opt.input.stem
    opt.output_directory = opt.output_directory / out_dir
    logger.info(f'Creating output directory : {opt.output_directory}')
    opt.output_directory.mkdir(parents=True, exist_ok=True) # make the output directory
    if opt.no_blur:
        logger.info("--no_blur is set")
    main(opt)
