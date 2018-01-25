import cv2

def get_frames(test_mp4_vod_path, save_path):
    video = cv2.VideoCapture(test_mp4_vod_path)
    print("Opened ", test_mp4_vod_path)
    print("Processing MP4 frame by frame")

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
            im = Image.fromarray(frame)#.crop((1625, 785, 1920, 1080))
            im = np.array(im, dtype = np.uint8)
            cv2.imwrite(BASE_DATA_PATH + "/%s/frames/frame_%d.jpg" %  (folder, file_count), im)
            file_count += 1
        count += 1

    print("Saved %d frames" % (file_count) )
    video.release()
