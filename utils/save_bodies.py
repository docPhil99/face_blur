import json
import numpy as np


def addIntoOutput(out, identifier, tab):
    out[identifier] = []
    for element in tab:
        out[identifier].append(element)
    return out

def serializeBodyData(body_data):
    """Serialize BodyData into a JSON like structure"""
    out = {}
    out["id"] = body_data.id
    out["unique_object_id"] = str(body_data.unique_object_id)
    out["tracking_state"] = str(body_data.tracking_state)
    out["action_state"] = str(body_data.action_state)
    addIntoOutput(out, "position", body_data.position)
    addIntoOutput(out, "velocity", body_data.velocity)
    addIntoOutput(out, "bounding_box_2d", body_data.bounding_box_2d)
    out["confidence"] = body_data.confidence
    addIntoOutput(out, "bounding_box", body_data.bounding_box)
    addIntoOutput(out, "dimensions", body_data.dimensions)
    addIntoOutput(out, "keypoint_2d", body_data.keypoint_2d)
    addIntoOutput(out, "keypoint", body_data.keypoint)
    addIntoOutput(out, "keypoint_cov", body_data.keypoints_covariance)
    addIntoOutput(out, "head_bounding_box_2d", body_data.head_bounding_box_2d)
    addIntoOutput(out, "head_bounding_box", body_data.head_bounding_box)
    addIntoOutput(out, "head_position", body_data.head_position)
    addIntoOutput(out, "keypoint_confidence", body_data.keypoint_confidence)
    addIntoOutput(out, "local_position_per_joint", body_data.local_position_per_joint)
    addIntoOutput(out, "local_orientation_per_joint", body_data.local_orientation_per_joint)
    addIntoOutput(out, "global_root_orientation", body_data.global_root_orientation)
    return out

def serializeBodies(bodies):
    """Serialize Bodies objects into a JSON like structure"""
    out = {}
    out["is_new"] = bodies.is_new
    out["is_tracked"] = bodies.is_tracked
    out["timestamp"] = bodies.timestamp.data_ns
    out["body_list"] = []
    for sk in bodies.body_list:
        out["body_list"].append(serializeBodyData(sk))
    return out

def _cam_pam(obj):
    out={}
    out["cx"] = obj.cx
    out["cy"] = obj.cy
    out["d_fov"] = obj.d_fov
    out["focal_length_metric"] = obj.focal_length_metric
    out["fx"] = obj.fx
    out["fy"] = obj.fy
    out["h_fov"] = obj.h_fov
    out["v_fov"] = obj.v_fov
    out["disto"] = obj.disto
    return out

def serializeConfig(config):
    out={}
    out["camera_model"] = str(config.camera_model)
    out["serial_number"] = config.serial_number
    out["fps"] = config.camera_configuration.fps
    out["resolution"] = [config.camera_configuration.resolution.width, config.camera_configuration.resolution.height]
    out["left_cam"] = _cam_pam(config.camera_configuration.calibration_parameters.left_cam)
    out["right_cam"] = _cam_pam(config.camera_configuration.calibration_parameters.left_cam)
    trans ={}
    trans["m"] = config.camera_configuration.calibration_parameters.stereo_transform.m
    trans["matrix_name"] = config.camera_configuration.calibration_parameters.stereo_transform.matrix_name
    out["stereo_transform"] = trans

    out["left_cam_raw"] = _cam_pam(config.camera_configuration.calibration_parameters_raw.left_cam)
    out["right_cam_raw"] = _cam_pam(config.camera_configuration.calibration_parameters_raw.left_cam)
    trans = {}
    trans["m"] = config.camera_configuration.calibration_parameters_raw.stereo_transform.m
    trans["matrix_name"] = config.camera_configuration.calibration_parameters_raw.stereo_transform.matrix_name
    out["stereo_transform_raw"] = trans

    return out

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

