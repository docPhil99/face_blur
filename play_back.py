import cv2
import numpy as np
import json
import argparse
from loguru import logger
from pathlib import Path
import gzip
import utils.viewer as gl

input_dir =Path('/root/Documents/ZED/ZED/ZED_Proc2/2025-03-26%2014.44.03%20recording%2034978846')
#
# #left camera
# path= input_dir/Path('left')
# for img_name in sorted(path.glob('*.png')):
#     logger.debug(f'Processing {img_name}')
#     img = cv2.imread(str(img_name))
#     cv2.imshow('left',img)
#     if cv2.waitKey(20) == ord('q'):
#         break
#
#
# # depth image
# path= input_dir/Path('depth')
# for img_name in sorted(path.glob('*.png')):
#     img = cv2.imread(str(img_name))
#     cv2.imshow('depth',img)
#     if cv2.waitKey(20) == ord('q'):
#         break
#
#
# # depth map
# path= input_dir/Path('depth')
# for img_name in sorted(path.glob('*.npy.gz')):
#     with gzip.GzipFile(img_name, "r") as f:
#         map =np.load(f)
#     max = np.max(map,where=~(np.isnan(map) | np.isinf(map)), initial=-1)
#     min  = np.min(map,where=~(np.isnan(map) | np.isinf(map)), initial=200)
#     norm_map = (map-min)/(max-min)
#     norm_map = np.nan_to_num(norm_map)
#
#     cv2.imshow('depth array',norm_map)
#     if cv2.waitKey(20) == ord('q'):
#         break

body_path= input_dir/Path('bodies.json')
with open(body_path) as f:
    bodies = json.load(f)

for frame_number in bodies.keys():
    print(frame_number)
    path = input_dir / Path('left')/Path(f'{int(frame_number):06}.png')
    img = cv2.imread(str(path))
    bods  = bodies[frame_number]["body_list"]
    for body in bods:
        kps = body["keypoint_2d"]
        for kp in kps:
            cv2.circle(img,(int(kp[0]),int(kp[1])),2,(255,0,0),2)

    cv2.imshow('body',img)
    if cv2.waitKey(20) == ord('q'):
        break

## 3D  - not working
# import pyzed.sl as sl
# import types
# with open(input_dir/Path('config.json'))as f:
#     config = json.load(f)
#
# conf = config["left_cam"]
# conf["width"] = config["resolution"][0]
# conf["height"] = config["resolution"][1]
# viewer = gl.GLViewer()
# viewer.init(conf,True, sl.BODY_FORMAT.BODY_34)
# #image = sl.Mat()
# while viewer.is_available():
#     for frame_number in bodies.keys():
#         print(frame_number)
#         bods=bodies[frame_number]
#         viewer.update_view(bods)
#         cv2.waitKey(10)
#
#
# viewer.exit()