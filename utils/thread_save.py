# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:39:46 2017

@author: phil
"""
from loguru import logger

#logger = logging.getLogger(__name__)
# import the necessary packages
from threading import Thread
import threading
import sys
from queue import Queue
import cv2



class FastVideoReader:
    def __init__(self, path, queueSize=128, colourspace='BGR'):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.colourspace = colourspace
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        # self._EOF=False
        self.event = threading.Event()
        self.event.clear()

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                print("stopped")
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file

                (grabbed, frame) = self.stream.read()

                if self.colourspace != 'BGR':
                    frame = cv2.cvtColor(frame, self.colourspace)
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return self.Q.get(block=True, timeout=2.0)

    def more(self):
        # return True if there are still frames in the queue
        return not self.stopped

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def skipFrames(self, number):
        # skips a number of frames fowards
        count = 0
        while self.more() and count < number:
            self.read()
            # cv2.waitKey(1)
            count += 1


class FVR():
    def __init__(self, filename):
        self.filename = filename
        self.fvr = FastVideoReader(filename)
        self.c = 0

    def __enter__(self):
        self.fvr.start()
        return self

    def read(self):
        self.c += 1
        print(self.c)
        while self.fvr.more():
            yield self.fvr.read()

        print("read failed")

    def more(self):
        return self.fvr.more()

    def __exit__(self, *args):
        self.fvr.stop()
        print("Exit called")

if __name__ == "__main__":
    # import imutils
    # import time
    #from imutils.video import FPS

    with FVR("/home/phil/Dropbox/Matlab/Videos/PVtest.mp4") as fv:
        for frame in fv.read():
            # while fv.more():
            # frame=fv.read()
            cv2.imshow("Frame", frame)
            res = cv2.waitKey(1)
    cv2.destroyAllWindows()
#
##    from imutils.video import FileVideoStream
#    #fvs=FileVideoStream("/home/phil/Dropbox/Matlab/Videos/Police/AJ11817.MPG").start()
#    fvs = FastVideoReader("/home/phil/Dropbox/Matlab/Videos/PVtest.mp4").start()
#  #  time.sleep(1.0)
#
#    # start the FPS timer
#    fps = FPS().start()
#    res = 0
#    #fvs.skipFrames(500)
#    while fvs.more() and not res==27:
#    	# grab the frame from the threaded video file stream, resize
#    	# it, and convert it to grayscale (while still retaining 3
#    	# channels)
#    	frame = fvs.read()
#    	#frame = imutils.resize(frame, width=450)
#    	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    	#frame = np.dstack([frame, frame, frame])
#
#    	# display the size of the queue on the frame
#    	cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
#    		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#    	# show the frame and update the FPS counter
#    	cv2.imshow("Frame", frame)
#    	res=cv2.waitKey(1)
#    	fps.update()
#    fps.stop()
#    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#
#    # do a bit of cleanup
#    cv2.destroyAllWindows()
#    fvs.stop()