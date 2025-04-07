

"""
    Read SVO and convert to images and depth array.
"""
import json

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
import utils.viewer as gl
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper



class face_blur:
    def __init__(self):
        pass


    @staticmethod
    @timeit
    def blur(image):
        resp = RetinaFace.detect_faces(image)
        bboxes =[]
        for key, face in resp.items():
            bbox = face['facial_area']
            bboxes.append(bbox)
            image = face_blur._blur_face(image, bbox)
        return image, bboxes

    @staticmethod
    def _blur_face(image, bbox):
        face = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        face_blur = cv2.blur(face, (200, 200))
        image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = face_blur
        return image
from utils.save_bodies import serializeBodies, NumpyEncoder,serializeConfig

class Body_Tracker:
    def __init__(self, camera, conf_threshold=40):
        self.skeleton_file_data = {}  # storage dict
        self.camera = camera
        self.body_params = body_params = sl.BodyTrackingParameters()
        # Different model can be chosen, optimizing the runtime or the accuracy
        self.body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        self.body_params.enable_tracking = True
        self.body_params.enable_segmentation = False
        # Optimize the person joints position, requires more computations
        self.body_params.enable_body_fitting = True
        self.body_params.body_format = sl.BODY_FORMAT.BODY_34
        # Object tracking requires the positional tracking module


        if self.body_params.enable_tracking:
            self.positional_tracking_param = sl.PositionalTrackingParameters()
            # positional_tracking_param.set_as_static = True
            self.positional_tracking_param.set_floor_as_origin = True
            self.camera.enable_positional_tracking(self.positional_tracking_param)

        err = self.camera.enable_body_tracking(body_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.error(f"Enable Body Tracking : {repr(err)}. Exit program.")
            self.camera.close()
            exit()
        self.bodies = sl.Bodies()
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        # For outdoor scene or long range, the confidence should be lowered to avoid missing detections (~20-30)
        # For indoor scene or closer range, a higher confidence limits the risk of false positives and increase the precision (~50+)
        self.body_runtime_param.detection_confidence_threshold = conf_threshold

    def save_data(self,dir_name:Path):
        filename = dir_name / Path('bodies.json')
        with open(filename, 'w') as f:
            json.dump(self.skeleton_file_data,f,indent=4,cls=NumpyEncoder)

    def draw2D(self,image):
        for body in self.bodies.body_list:
            kp = body.keypoint_2d
            for point in kp:
                cv2.circle(image,(int(point[0]),int(point[1])), 3,(255,0,0),-1)

        #kp = self.bodies
    def process_frame(self,frame_num):
        err = self.camera.retrieve_bodies(self.bodies, self.body_runtime_param)
        self.skeleton_file_data[frame_num] = serializeBodies(self.bodies)
        # if self.bodies.is_new:
        #     body_array = self.bodies.body_list
        #     print(str(len(body_array)) + " Person(s) detected\n")
        #     if len(body_array) > 0:
        #         first_body = body_array[0]
        #         print("First Person attributes:")
        #         print(" Confidence (" + str(int(first_body.confidence)) + "/100)")
        #         if self.body_params.enable_tracking:
        #             print(" Tracking ID: " + str(int(first_body.id)) + " tracking state: " + repr(
        #                 first_body.tracking_state) + " / " + repr(first_body.action_state))
        #         position = first_body.position
        #         velocity = first_body.velocity
        #         dimensions = first_body.dimensions
        #         print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(
        #             position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], dimensions[0],
        #             dimensions[1], dimensions[2]))
        #         if first_body.mask.is_init():
        #             print(" 2D mask available")
        #
        #         print(" Keypoint 2D ")
        #         keypoint_2d = first_body.keypoint_2d
        #         for it in keypoint_2d:
        #             print("    " + str(it))
        #         print("\n Keypoint 3D ")
        #         keypoint = first_body.keypoint
        #         for it in keypoint:
        #             print("    " + str(it))



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
        self.config_save_path = self.opt.output_directory/Path('config.json')
        self.left_image_path.mkdir(parents=True, exist_ok=True)
        self.right_image_path.mkdir(parents=True, exist_ok=True)
        self.depth_image_path.mkdir(parents=True, exist_ok=True)

        self._open_file()
        self.body_tracker = Body_Tracker(self.cam)
        if self.opt.show_3D:
            self.viewer= gl.GLViewer()
            self.viewer.init()

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
        resolution = self.cam.get_camera_information().camera_configuration.resolution
        config = self.cam.get_camera_information()


        with open(self.config_save_path ,'w') as f:
            config_dict = serializeConfig(config)
            json.dump(config_dict, f, indent=4, cls=NumpyEncoder)


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
        self.face_bbox={}

        logger.info(f"SVO contains {self.nb_frames}  frames at {self.svo_frame_rate} fps")

    def _save_image(self,filename,image, blur=False):
        logger.debug(f"Saved image :  {filename}")
        img = image.get_data()
        #drop alpha
        img = img[:,:,0:3]
        if blur:
            img,bbox = face_blur.blur(img)
            self.face_bbox[filename.stem]=bbox

        cv2.imwrite(str(filename),img)

    @timeit
    def _save_images(self):

        #left
        filename = self.left_image_path/Path(f'{self.svo_position:06}.png')
        self._timestamps[f'{self.svo_position:06}.png']=self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        self._save_image(filename,self.left_image,blur=not self.opt.no_blur)

        filename = self.right_image_path / Path(f'{self.svo_position:06}.png')
        self._save_image(filename,self.right_image,blur=not self.opt.no_blur)
        # depth
        filename = self.depth_image_path / Path(f'{self.svo_position:06}.png')
        self._save_image(filename,self.depth_image,blur=False)

        if not opt.no_depth:
            # depth map
            filename = self.depth_image_path / Path(f'{self.svo_position:06}.npy.gz')
            try:
                with gzip.GzipFile(filename,"w") as f:
                    np.save(file=f, arr=self.depth_map.get_data())
            except Exception as e:
                logger.exception(f'Could not save {filename}')
            # point cloud
            filename = self.depth_image_path / Path(f'{self.svo_position:06}.ply')
            tmp = sl.Mat()
            self.cam.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
            saved = (tmp.write(str(filename)) == sl.ERROR_CODE.SUCCESS)
            if not saved:
                logger.error(f"Failed to write {filename}. Please check that you have permissions to write on disk")
    def process_loop(self):
        self.run_flag = True
        while self.run_flag:
            start_time = time.perf_counter()
            err = self.cam.grab(self.runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self.svo_position = self.cam.get_svo_position()
                self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT)
                self.cam.retrieve_image(self.right_image, sl.VIEW.RIGHT)
                self.cam.retrieve_image(self.depth_image, sl.VIEW.DEPTH)
                self.cam.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                self._save_images()
                self.body_tracker.process_frame(self.svo_position)
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
            logger.debug(f"Frame time: {total_time} seconds")
        self.body_tracker.save_data(opt.output_directory)
        with open(self.opt.output_directory/Path('timestamps.json'),'wt') as f:
            json.dump(self._timestamps,f)
        with open(self.opt.output_directory/Path('faces.json'),'wt') as f:
            json.dump(self.face_bbox,f,cls=NumpyEncoder)

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
    proc.process_loop()
    proc.shutdown()



def _proc1(opt):
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
    opt.output_directory.mkdir(parents=True, exist_ok=True)  # make the output directory

    main(opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--input', '-i', type=Path, help='Path to the SVO file')
    parser.add_argument('--output_directory', '-o', type=Path, help='Path to the output directory', required=True)
    parser.add_argument('--no_blur', '-n', action='store_true', help="Don't blur the faces")
    parser.add_argument('--no_depth', action='store_true', help="Don't store depth")
    parser.add_argument('--show_3D', action='store_true', help="Display 3D bodies")
    parser.add_argument('--no_display', action='store_true', help="Don't display images")
    opt = parser.parse_args()
    if opt.no_blur:
        logger.info("--no_blur is set")

    logger.info(f"Processing single file {opt.input}")
    _proc1(opt)
