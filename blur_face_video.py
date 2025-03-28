from retinaface import RetinaFace
import cv2
import argparse
from pathlib import Path
from loguru import logger
import sys


def blur_face(image, bbox):
    face = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    face_blur = cv2.blur(face, (200, 200))
    image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = face_blur
    # cv2.imshow('face',face_blur)
    return image

parser = argparse.ArgumentParser()
parser.add_argument('--input','-i', type=Path, required=True,  help="input video file")
group = parser.add_mutually_exclusive_group()
group.add_argument('--output','-o', type=Path, required=False,  help="output file name. If not set appends _blur to the input file name")
group.add_argument('--output_dir',type=Path, required=False, help="output directory, uses input file with _blur appened")
parser.add_argument('--fps',type=float,help='override output file fps setting', required=False)
parser.add_argument('--fourcc',type=str,help='fourcc code, defaults to mp4v',default='mp4v')

args = parser.parse_args()



vid_path = args.input
if args.output_dir:    # write file to output directory
    ext = vid_path.suffix
    output_path = args.output_dir/Path(f'{vid_path.stem}_blur{ext}')
    output_path.parents[0].mkdir(parents=True, exist_ok=True) # make the directory if it doesn't exist.
elif args.output :    # user has specified path
    output_path = args.output
else:  # no file name, write video file to input directory
    par = vid_path.parents[0]
    ext = vid_path.suffix
    output_path = par/Path(f'{vid_path.stem}_blur{ext}')

logger.info(f'output path: {output_path}')


video_capture = cv2.VideoCapture(str(vid_path))
video_writer = None
if (video_capture.isOpened() == False):
    logger.error(f"Error opening video {vid_path}")
    sys.exit(1)

if args.fps:
    fps = args.fps
else:
    fps = video_capture.get(cv2.CAP_PROP_FPS)

frame_number =0
while (video_capture.isOpened()):
    # Capture each frame
    ret, face_image = video_capture.read()
    if ret:
        if not video_writer:
            fcc = args.fourcc
            logger.info(f'using {fcc}')
            video_writer = cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*fcc),fps,face_image.shape[1::-1] )
        #cv2.imshow('org',face_image)
        resp = RetinaFace.detect_faces(face_image)
        frame_number+=1
        print(f"Frame {frame_number}\n{resp}")
        for key, face in resp.items():
            bbox = face['facial_area']
            face_image = blur_face(face_image, bbox)
        video_writer.write(face_image)
        cv2.imshow('img', face_image)
        if cv2.waitKey(10) == ord('q'):
            break
video_writer.release()
video_capture.release()

cv2.destroyAllWindows()
