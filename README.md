# faceBlur

Blur faces in video files

## blur_face_video.py

This is a general face blurrer for video only. Not svo files

### Install

The code uses rentina-face to detect faces. Ideally this wants to run on a CUDA GPU.

`pip install -r requirements.txt`

should work, however this depends on the correct CUDA installed on your system.

```
conda create -n face_blur python==3.10 -y 
conda activate face_blur
conda install tensorflow-gpu
conda install -c conda-forge retina-face
pip install loguru
```

## SVO face blur

Try `pip install -r requirements.txt`
Then install the ZED SDK.

Run `python blur_face_svo.py --input your_video.svo2 --output_dir out_directory`
This will create a directory _out_directory_ and within that a directory based on the filename of
the input file. The images are contained with _left_ and _right_ directories. _depth_ contains a 
image version of the depth (not very useful), a gzipped numpy array of the depth and a point cloud for 
each image.
