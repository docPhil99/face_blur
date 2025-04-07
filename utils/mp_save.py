from time import sleep
from random import random
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
from pathlib import Path
import cv2


def _mp_save( queue):
    print('Consumer: Running', flush=True)
    # consume work
    while True:
        # get a unit of work
        item = queue.get()
        #sleep(2)
        # check for stop
        if item is None:
            break
        filename, img = item
        cv2.imwrite(str(filename), img)
        # report
        print(f'>got item', flush=True)
    # all done
    print('Consumer: Done', flush=True)

class CV_Image_Save:
    def __init__(self):
        self.queue = Queue(maxsize=1)

    def start(self):
        self.consumer_process = Process(target=_mp_save,args=(self.queue,))
        self.consumer_process.start()

    def save_image(self, img, filename :Path):
        self.queue.put((filename,img))



    def stop(self):
        self.queue.put(None)
        self.consumer_process.join()


# generate work
def producer():
    print('Producer: Running', flush=True)
    mp_save = CV_Image_Save()
    mp_save.start()
    # generate work
    for i in range(10):
        # generate a value
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        filename = Path(f'tmp/f{i}.png')
        # block
        print('Writing', filename, flush=True)
        # add to the queue
        #queue.put((filename, img))
        mp_save.save_image(img, filename)
    # all done
    mp_save.stop()
    #queue.put(None)
    print('Producer: Done', flush=True)


# consume work
def cv2_image_save(queue):
    print('Consumer: Running', flush=True)
    # consume work
    while True:
        # get a unit of work
        item = queue.get()
        sleep(2)
        # check for stop
        if item is None:
            break
        filename, img = item
        cv2.imwrite(str(filename), img)
        # report
        print(f'>got item', flush=True)
    # all done
    print('Consumer: Done', flush=True)


# entry point
if __name__ == '__main__':

    producer()
