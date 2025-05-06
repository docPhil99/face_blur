import cv2
import numpy as np
import json
from pathlib import Path
from loguru import logger

input_dir =Path('/home/d_phil/intentMAPS/ZED/ZED_Processed/BLACKB4P14S')

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
try:
    with open(body_path) as f:
        bodies = json.load(f)
except:
    logger.exception(f'Failed to open {body_path}')
    exit(-1)
for frame_number in bodies.keys():
    print(frame_number)
    path = input_dir / Path('left')/Path(f'{int(frame_number):06}.png')
    limg = cv2.imread(str(path))
    bods  = bodies[frame_number]["body_list"]
    for body in bods:
        kps = body["keypoint_2d"]
        for kp in kps:
            cv2.circle(limg,(int(kp[0]),int(kp[1])),2,(255,0,0),2)
    path = input_dir / Path('right') / Path(f'{int(frame_number):06}.png')
    rimg = cv2.imread(str(path))

    path = input_dir / Path('depth') / Path(f'{int(frame_number):06}.png')
    dimg = cv2.imread(str(path))
    # concatenate image Horizontally
    sz = limg.shape
    img = np.concatenate((limg, rimg, dimg), axis=1)
    img = cv2.resize(img, (sz[1],sz[0]//3))
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