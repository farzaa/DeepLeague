# takes a VOD, and outputs the frames. frames to campture per second can be adjusted.
# this program was made to run manually

import cv2
from PIL import Image
import numpy as np
import os
from paths import BASE_DATA_PATH

def get_frames():
    for folder in os.listdir(BASE_DATA_PATH):
        print("Currently on ", folder)
        if os.path.isdir(BASE_DATA_PATH + folder):
            items = os.listdir(BASE_DATA_PATH + folder)
            if 'frames' not in items:
                print("Creating frames dir... ")
                os.mkdir(BASE_DATA_PATH + folder + '/frames')
            # skip if frames are already done for that folder
            else:
                print("Frames already exists. skipping...")
                continue
        # skip non-folders
        else:
            continue
        video = cv2.VideoCapture(BASE_DATA_PATH + '%s/vod.mp4' % folder)

        # forward over to the frames you want to start reading from.
        # manually set this, fps * time in seconds you wanna start from
        video.set(1, 0);
        success, frame = video.read()
        count = 0
        file_count = 0
        success = True
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Loading video %d seconds long with FPS %d and total frame count %d " % (total_frame_count/fps, fps, total_frame_count))

        while success:
            success, frame = video.read()
            if not success:
                break
            if count % 1000 == 0:
                print("Currently at frame ", count)

            # i save once every fps, which comes out to 1 frames per second.
            # i think anymore than 2 FPS leads to to much repeat data.
            if count %  fps == 0:
                im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(im)#.crop((1625, 785, 1920, 1080))
                im = np.array(im, dtype = np.uint8)
                cv2.imwrite(BASE_DATA_PATH + "/%s/frames/frame_%d.jpg" %  (folder, file_count), im)
                file_count += 1
            count += 1

        print("Saved %d frames" % (file_count) )
        video.release()

if __name__ == "__main__":
    get_frames()
