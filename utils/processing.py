from utils.timeit import timeit
import cv2
from retinaface import RetinaFace
import pyzed.sl as sl
from loguru import logger
import json
from pathlib import Path
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
from utils.serialiser import serializeBodies, NumpyEncoder,serializeConfig

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