"""
Demonstrates how to play back the images
"""


import cv2
import numpy as np
import json
from pathlib import Path
from loguru import logger
from utils import sort
import argparse

def _blur_face(image, bbox,scale=1):
    bbox[0]=bbox[0]-scale
    bbox[1]=bbox[1]-scale
    bbox[2]=bbox[2]+scale
    bbox[3]=bbox[3]+scale
    bbox = [int(x) for x in bbox]
    face = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    if face.shape[0] == 0 or face.shape[1] == 0:
        logger.debug(face.shape)
        return image
    face_blur = cv2.blur(face, (200, 200))
    image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = face_blur
    return image
def draw_trackers(img, dets, trackers):

    tracker_dets = [t.get_state()[0] for t in trackers]
    tracker_dets = [t.astype(int).tolist() for t in tracker_dets]
    dets = dets.astype(int).tolist()

    #for det in dets:
    #    img = cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), (255, 255, 0), 2)

    for det in tracker_dets:
        img = cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), (255, 255, 255), 2)
    return img




def process(input_dir,save_path=None,left_only=False,no_display=False):
    config_path = input_dir/Path('config.json')
    try:
        with open(config_path) as f:
            config = json.load(f)
    except:
        logger.exception(f'Failed to open {config_path}')
        exit(-1)

    body_path= input_dir/Path('bodies.json')
    try:
        with open(body_path) as f:
            bodies = json.load(f)
    except:
        logger.exception(f'Failed to open {body_path}')
        exit(-1)

    face_path = input_dir/Path('right_faces.json')
    try:
        with open(face_path) as f:
            right_faces = json.load(f)
    except:
        logger.exception(f'Failed to open {face_path}, trying old name')
        face_path = input_dir / Path('faces.json')
        try:
            with open(face_path) as f:
                right_faces = json.load(f)
        except:
            logger.exception(f'Failed to open {face_path}')
            exit(-1)

    face_path = input_dir/Path('left_faces.json')
    try:
        with open(face_path) as f:
            left_faces = json.load(f)
    except:
        logger.exception(f'Failed to open {face_path}')
        left_faces = None

    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vid_size = config['resolution']
        logger.info(f'Opening video writer at {save_path}, resolution: {vid_size}, fps: {config["fps"]}')
        vid_writer= cv2.VideoWriter(str(save_path), fourcc, config['fps'], vid_size)
    max_frame_number = max([int(x) for x in bodies.keys()])
    logger.debug(f'Max frame number: {max_frame_number}')
    frame_number = 0

    r_face_tracker = sort.Sort(max_age=5)
    l_face_tracker = sort.Sort(max_age=5)

    kp_tracker = sort.Sort(max_age=5)

    #trans_m=np.array(config['stereo_transform_raw']['m'])
    #trans_m=np.linalg.inv(trans_m)

    while True:
        print(frame_number)
        path = input_dir / Path('left')/Path(f'{int(frame_number):06}.png')
        limg = cv2.imread(str(path))
        if not left_only:
            path = input_dir / Path('right') / Path(f'{int(frame_number):06}.png')
            rimg = cv2.imread(str(path))
        try:
            bods  = bodies[str(frame_number)]["body_list"]
            all_left_faces = []
            for body in bods:
                kps = body["keypoint_2d"]

                left = [0,0]
                right  = [0,0]
                left[0]=min([k[0] for k in kps[26:31]])
                left[1]=min([k[1] for k in kps[26:31]])
                right[0]=max([k[0] for k in kps[26:31]])
                right[1] = max([k[1] for k in kps[26:31]])
                all_left_faces.append([left[0],left[1],right[0],right[1],1])
                limg=_blur_face(limg,[left[0],left[1],right[0],right[1]],scale=4.0)
                #cv2.rectangle(limg,(int(left[0]), int(left[1])), (int(right[0]), int(right[1])),  (0, 255, 0), 3)
            #kp_dets, kp_tracks = kp_tracker.update(np.array(all_left_faces))
            #limg=draw_trackers(limg,kp_dets,kp_tracks)
        except KeyError as e:
            logger.error(f'No body detected: {e}')


        #draw face rects
        f_key =f'{frame_number:06}'
        right_face_list = right_faces[f_key]
        #convert to sort
        [x.append(1) for x in right_face_list]
        if left_faces:
            left_face_list = left_faces[f_key]
            [x.append(1) for x in right_face_list]
            l_dets, l_trackers = l_face_tracker.update(np.array(left_face_list))
        #print(left_face_list)
        r_dets, r_trackers = r_face_tracker.update(np.array(right_face_list))


        #if left_faces:
            #for bbox in left_face_list:
            #    limg = cv2.rectangle(limg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        #    limg = draw_trackers(limg,l_dets, l_trackers)

        #for bbox in right_face_list:
            #rimg = cv2.rectangle(rimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        if not left_only:
            rimg = draw_trackers(rimg, r_dets, r_trackers)
            # concatenate image Horizontally
            sz = limg.shape
            img = np.concatenate((limg, rimg), axis=1)
            #img = cv2.resize(img, (sz[1],sz[0]//2))
            if not no_display:
                cv2.imshow('body',img)
        else:
            if not no_display:
                cv2.imshow('body',limg)
        if save_path:

            vid_writer.write(limg)
            frame_number += 1
            if frame_number == max_frame_number:
                break
            if not no_display:
                key = cv2.waitKey(10)
                if key == ord('q'):
                    break

        else:
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            if key == ord('x') and frame_number < max_frame_number:
                frame_number= frame_number+1
            if key == ord('z') and frame_number > 0:
                frame_number = frame_number-1

    if save_path:
        vid_writer.release()
        logger.info('Closed video writer')
#input_dir =Path('/home/d_phil/intentMAPS/ZED/ZED_Proc_new_test/2025-03-26 14.46.59 recording 39493447')
#input_dir =Path('/home/d_phil/intentMAPS/test_set/processed/BLACKB4P14S')

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--input',type=Path,help='Path to the video directory',required=True)
    args.add_argument('--no_display',action='store_true')
    args.add_argument('--save',action='store_true',help='Save video')
    args.add_argument('--save_path',type=Path,help='Path to the saved video directory, if not set, it uses the input directory')
    opt = args.parse_args()
    input_dir = opt.input
    logger.info(f'Input path: {input_dir}')
    if opt.save:
        if opt.save_path is None:
            save_path = input_dir/Path(f'{input_dir.stem}_left.mp4')
        else:
            opt.save_path.mkdir(parents=True, exist_ok=True)
            save_path = opt.save_path/Path(f'{input_dir.stem}_left.mp4')
        logger.info(f'Output path: {save_path}')
    else:
        save_path = None
    process(input_dir,save_path=save_path, left_only=True, no_display=opt.no_display)