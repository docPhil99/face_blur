# faceBlur

Blur faces in SVO video files


## SVO face blur

Try `pip install -r requirements.txt`
Then install the ZED SDK.

Run `python blur_face_svo.py --input your_video.svo2 --output_dir out_directory`
This will create a directory _out_directory_ and within that a directory based on the filename of
the input file. The images are contained with _left_ and _right_ directories. _depth_ contains a 
image version of the depth (not very useful), a gzipped numpy array of the depth and a point cloud for 
each image.

There is a chance that RetinaFace will miss a face, so the video is saved as .png images. They can be converted to video 
with ffmpeg. Edit as needed:

`ffmpeg -framerate 30 -pattern_type glob -i 'left/*.png' -c:v libx264 -pix_fmt yuv420p left.mp4`

Note: the outputted file `config.json` contains the actual frame rate data if needed.

## Install

The code uses rentina-face to detect faces. Ideally, this wants to run on a CUDA GPU.

`pip install -r requirements.txt`

should work, however, this depends on the correct CUDA installed on your system. So conda might be needed.

```
conda create -n face_blur python==3.10 -y 
conda activate face_blur
conda install tensorflow-gpu
conda install -c conda-forge retina-face
pip install loguru
```

you might need to install the requirements.txt after this.

## Docker

To run in docker the mounts need explaining. 
Edit the directories before the : to match your system.

```
        volumes:
            - /dev:/dev   # needed for the camera to work
            - /tmp/.X11-unix:/tmp/.X11-unix   #needed for gui
            - ~/.Xauthority:/root/.Xauthority #needed for gui
            - ~/python/IntentMAPS/ZED/resources:/usr/local/zed/resources  # where the ZED SDK stores its weight files
            - ~/python/IntentMAPS:/home/d_phil/intentMAPS # location of the this repo
            - /mnt/intentMAPS/intentMAPS/test_set:/home/d_phil/intentMAPS/test_set  # optional location of data, if you don't want it to be a subdirectory of the repo.
            
```