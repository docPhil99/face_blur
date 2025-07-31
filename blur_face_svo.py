

"""
    Read SVO and convert to images and depth array.
"""
import json


import sys
import pyzed.sl as sl
import cv2
import argparse
import os
from loguru import logger
from pathlib import Path
import numpy as np
import gzip
import utils.viewer as gl
from utils.timeit import timeit
import time
import datetime
from utils.processing import Body_Tracker, face_blur
from utils.serialiser import serializeConfig, NumpyEncoder





class SVO_Process:
    def __init__(self,opt):
        self.opt = opt
        self.filepath = opt.input
        self.cam = None
        self.svo_image =None
        self.runtime = None
        self.left_image_path = None
        self.right_image_path = None
        self.depth_image_path = None
        self.config_save_path = None

        self._open_file()
        self.body_tracker = Body_Tracker(self.cam)
        if self.opt.show_3D:
            self.viewer= gl.GLViewer()
            self.viewer.init()

    def _set_output_paths(self):
        timestamp = self._timestamps[f'{0:06}.png']
        date_dir = Path(datetime.datetime.fromtimestamp(timestamp/1000).strftime('%d_%m_%Y'))
        out_dir = opt.input.stem
        opt.output_directory = opt.output_directory /date_dir / out_dir
        logger.info(f'Creating output directory : {opt.output_directory}')
        try:
            opt.output_directory.mkdir(parents=True, exist_ok=self.opt.overwrite)  # make the output directory
        except FileExistsError:
            logger.error(f'Output directory {opt.output_directory} already exists. Use --overwrite option to overwrite. Exiting.')
            self.shutdown()
            sys.exit(1)

        self.left_image_path = self.opt.output_directory / Path('left')
        self.right_image_path = self.opt.output_directory  / Path('right')
        self.depth_image_path = self.opt.output_directory  / Path('depth')
        self.config_save_path = self.opt.output_directory / Path('config.json')
        self.left_image_path.mkdir(parents=True, exist_ok=True)
        self.right_image_path.mkdir(parents=True, exist_ok=True)
        self.depth_image_path.mkdir(parents=True, exist_ok=True)

    def _save_config_file(self):
        with open(self.config_save_path, 'w') as f:
            logger.info(f'Saving {self.config_save_path}')
            config = self.cam.get_camera_information()
            config_dict = serializeConfig(config)
            json.dump(config_dict, f, indent=4, cls=NumpyEncoder)
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
        init.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
        #init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        self.cam = sl.Camera()

        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera opened succesfully
            logger.error(f"Camera Open {status}.")
            raise Exception(f"Camera File Not Open: {status}")

        # Set a maximum resolution, for visualisation confort
        #resolution = self.cam.get_camera_information().camera_configuration.resolution






        self.runtime = sl.RuntimeParameters()
        # Prepare single image containers
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        self.depth_image = sl.Mat()
        self.depth_map = sl.Mat()
        self.mat = sl.Mat()
        self.svo_frame_rate = self.cam.get_init_parameters().camera_fps
        self.nb_frames = self.cam.get_svo_number_of_frames()
        self._timestamps ={}
        self.right_face_bbox={}
        self.left_face_bbox={}

        logger.info(f"SVO contains {self.nb_frames}  frames at {self.svo_frame_rate} fps")

    @timeit
    def _save_image(self,filename,image, blur=False,cam='right' ):
        #logger.debug(f"Saved image :  {filename}")
        img = image.get_data()
        #drop alpha
        img = img[:,:,0:3]
        img = np.ascontiguousarray(img) # some opencv function need this.
        if blur:
            img,bbox = face_blur.blur(img)
            img = self.body_tracker.blur_face_regions(img, cam)
            if cam=='left':
                self.left_face_bbox[filename.stem] = bbox
            else:
                self.right_face_bbox[filename.stem]=bbox

        cv2.imwrite(str(filename),img)


    @timeit
    def _save_depth(self):
        filename = self.depth_image_path / Path(f'{self.svo_position:06}.npy.gz')
        try:
            with gzip.GzipFile(filename, "w") as f:
                np.save(file=f, arr=self.depth_map.get_data())
        except Exception as e:
            logger.exception(f'Could not save {filename}')

    @timeit
    def _save_point_cloud(self):
        filename = self.depth_image_path / Path(f'{self.svo_position:06}.{self.opt.point_cloud_extension}')
        tmp = sl.Mat()
        self.cam.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
        saved = (tmp.write(str(filename)) == sl.ERROR_CODE.SUCCESS)
        if not saved:
            logger.error(f"Failed to write {filename}. Please check that you have permissions to write on disk")
    @timeit
    def _save_images(self):


        #left
        self._timestamps[f'{self.svo_position:06}.png']=self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        if self.svo_position==0:  #first frame
            self._set_output_paths()
            self._save_config_file()

        filename = self.left_image_path / Path(f'{self.svo_position:06}.png')
        self._save_image(filename,self.left_image,blur=not self.opt.no_blur, cam='left')

        if self.opt.left_only:
            return

        filename = self.right_image_path / Path(f'{self.svo_position:06}.png')
        self._save_image(filename,self.right_image,blur=not self.opt.no_blur, cam='right')
        # depth
        #filename = self.depth_image_path / Path(f'{self.svo_position:06}.png')
        #self._save_image(filename,self.depth_image,blur=False)

        if not opt.no_depth:
            # depth map
            self._save_depth()
        if not opt.no_point_cloud:
            # point cloud
            self._save_point_cloud()

    def process_loop(self):
        self.run_flag = True
        while self.run_flag:
            start_time = time.perf_counter()
            err = self.cam.grab(self.runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self.svo_position = self.cam.get_svo_position()
                self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT)
                if not self.opt.left_only:
                    self.cam.retrieve_image(self.right_image, sl.VIEW.RIGHT)
                    self.cam.retrieve_image(self.depth_image, sl.VIEW.DEPTH)
                    self.cam.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                    self.body_tracker.process_frame(self.svo_position)

                self._save_images()

                cv_img = self.left_image.get_data()
                if self.opt.show_3D:
                    self.viewer.update_bodies(self.body_tracker.bodies)
                self.body_tracker.draw2D(cv_img)

                if not opt.no_display:
                    cv2.imshow("ViewL", cv_img)
                    key = cv2.waitKey(10)
                    if key == ord('q'):
                        self.run_flag = False

                self.progress_bar(self.svo_position/ self.nb_frames * 100, 30)
            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:  # Check if the .svo has ended
                self.progress_bar(100, 30)
                logger.info("SVO end has been reached.")
                self.run_flag = False
            else:
                logger.error("Grab ZED : ", err)
                self.run_flag = False
            end_time = time.perf_counter()
            total_time = end_time - start_time
            logger.debug(f"Frame time: {total_time} seconds for frame {self.svo_position}")
        if not self.opt.left_only:
            self.body_tracker.save_data(opt.output_directory)
        with open(self.opt.output_directory/Path('timestamps.json'),'wt') as f:
            json.dump(self._timestamps,f)
        if not opt.no_blur:
            with open(self.opt.output_directory/Path('left_faces.json'),'wt') as f:
                json.dump(self.left_face_bbox,f,cls=NumpyEncoder)
            if not opt.left_only:
                with open(self.opt.output_directory / Path('right_faces.json'), 'wt') as f:
                    json.dump(self.right_face_bbox, f, cls=NumpyEncoder)

    def shutdown(self):
        if opt.show_3D:
            self.viewer.exit()
        cv2.destroyAllWindows()
        self.cam.disable_body_tracking()
        self.cam.disable_positional_tracking()
        self.left_image.free()
        self.right_image.free()
        self.depth_image.free()
        self.depth_map.free()
        self.mat.free()
        self.cam.close()


def main(opt):
    proc  = SVO_Process(opt)
    if not opt.calib_only:
        proc.process_loop()
    proc.shutdown()


@timeit
def _proc1(opt):
    if not opt.input.suffix==".svo" and not opt.input.suffix==".svo2":
        print("--input_svo_file parameter should be a .svo file but is not : ", opt.input, "Exit program.")
        exit()
    if not os.path.isfile(opt.input):
        print("--input_svo_file parameter should be an existing file but is not : ", opt.input,
              "Exit program.")
        exit()


    main(opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--input', '-i', type=Path, help='Path to the SVO file',required=True)
    parser.add_argument('--output_directory', '-o', type=Path, help='Path to the output directory', required=True)
    parser.add_argument('--no_blur', '-n', action='store_true', help="Don't blur the faces")
    parser.add_argument('--no_depth', action='store_true', help="Don't store depth")
    parser.add_argument('--show_3D', action='store_true', help="Display 3D bodies")
    parser.add_argument('--no_display', action='store_true', help="Don't display images")
    parser.add_argument('--no_point_cloud', action='store_true', help="Don't save point cloud")
    parser.add_argument('--point_cloud_extension','-p',type=str,default='.ply',help="Extension of point cloud files")
    parser.add_argument('--left_only', action='store_true', help="only process the left image, no depth, no point cloud, no tracking")
    parser.add_argument('--calib_only', action='store_true', help="only extract calib.json")
    parser.add_argument('--overwrite', action='store_true', help="overwrite existing files")
    opt = parser.parse_args()
    logger_file = Path('logs') / Path(f'{opt.input.name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    logger_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(logger_file, level="DEBUG")
    if opt.no_blur:
        logger.info("--no_blur is set")
    if opt.left_only:
        opt.no_depth = True
        opt.no_point_cloud = True
        opt.show_3D = False
    logger.info(opt)
    logger.info(f"Processing single file {opt.input}")

    _proc1(opt)
