#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
from subprocess import call
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

import cv2

from yad2k.models.keras_yolo import yolo_eval, yolo_head

from retrain_yolo import create_model

import youtube_dl

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on the LoL minimap. Choose')

parser.add_argument(
    '--model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model',
    default='model_data/yolo.h5')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/league_classes.txt')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)

parser.add_argument(
    '-out',
    '--output_path',
    type=str,
    help='path to output test images')

parser.add_argument(
    '-champs',
    '--champs_in_game',
    type=str,
    help='to help avoid bad predictions, tell DeepLeague the 10 champions in the game of the VOD you are passing',
    default= "")

subparsers = parser.add_subparsers(dest='subcommand')

youtube_option = subparsers.add_parser('youtube')
youtube_option.add_argument(
    '-yt',
    '--test_youtube_link',
    type=str,
    help='a YouTube link to the VOD you want to analyze. Note - only 1080p videos are allowed!')
youtube_option.add_argument(
    '-yt_path',
    '--youtube_download_path',
    type=str,
    help='path to download YouTube video to')

youtube_option.add_argument(
    '-start',
    '--start_time',
    type=str,
    help='time when the game starts in the actual VOD. input in the format HH:MM:SS. ex. for 1:30 type 00:01:30'
)
youtube_option.add_argument(
    '-end',
    '--end_time',
    type=str,
    help='time when the game starts in the actual VOD. input in the format HH:MM:SS. ex. for 1:30 type 00:01:30'
)

vod_option = subparsers.add_parser('mp4')
vod_option.add_argument(
    '-mp4',
    '--test_mp4_vod_path',
    type=str,
    help='path to VOD to analyze. Note - only 1080p videos are allowed!'
)

image_option = subparsers.add_parser('images')
image_option.add_argument(
    '-images',
    '--test_images_path',
    help='path to images to test. These images MUST be size 1920x1080')

npz_option = subparsers.add_parser('npz')
npz_option.add_argument(
    '-npz',
    '--test_npz_path',
    help='path to npz file to test with image/bounding box objects. see GitHub for a download link.')

args = parser.parse_args()

model_path = os.path.expanduser(args.model_path)
assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
anchors_path = os.path.expanduser(args.anchors_path)
classes_path = os.path.expanduser(args.classes_path)
output_path = os.path.expanduser(args.output_path)

champs_in_game = os.path.expanduser(args.champs_in_game)

user_did_specify_champs = False

if champs_in_game != "":
    user_did_specify_champs = true
    champs_in_game = champs_in_game.split(" ")

if args.subcommand == 'images':
    test_images_path = os.path.expanduser(args.test_images_path)

if args.subcommand == 'npz':
    test_npz_path = os.path.expanduser(args.test_npz_path)

if args.subcommand == 'mp4':
    test_mp4_vod_path = os.path.expanduser(args.test_mp4_vod_path)


if args.subcommand == 'youtube':
    test_youtube_link = os.path.expanduser(args.test_youtube_link)
    youtube_download_path = os.path.expanduser(args.youtube_download_path)
    start_time = os.path.expanduser(args.start_time)
    end_time = os.path.expanduser(args.end_time)


if not os.path.exists(output_path):
    print('Creating output path {}'.format(output_path))
    os.mkdir(output_path)

sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

# yolo_model = load_model(model_path)
yolo_model, _ = create_model(anchors, class_names)
yolo_model.load_weights('trained_stage_3_best.h5')

# Verify model, anchors, and classes are compatible
num_classes = len(class_names)
num_anchors = len(anchors)
# TODO: Assumes dim ordering is channel last
model_output_channels = yolo_model.layers[-1].output_shape[-1]
assert model_output_channels == num_anchors * (num_classes + 5), \
    'Mismatch between model and given anchor and class sizes. ' \
    'Specify matching anchors and classes with --anchors_path and ' \
    '--classes_path flags.'
print('{} model, anchors, and classes loaded.'.format(model_path))

# Check if model is fully convolutional, assuming channel last order.
model_image_size = yolo_model.layers[0].input_shape[1:3]
is_fixed_size = model_image_size != (None, None)

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

# Generate output tensor targets for filtered bounding boxes.
# TODO: Wrap these backend operations with Keras layers.
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(
    yolo_outputs,
    input_image_shape,
    score_threshold=args.score_threshold,
    iou_threshold=args.iou_threshold)

def test_yolo(image, image_file_name):
    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = image.resize(
            tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        print(image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} boxes for {}'.format(len(out_boxes), image_file_name))

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        if user_did_specify_champs and predicted_class not in champs_in_game:
            continue

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    image.save(os.path.join(output_path, image_file_name), quality=90)

def process_mp4(test_mp4_vod_path):
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
            im = Image.fromarray(im).crop((1625, 785, 1920, 1080))
            test_yolo(im, str(file_count) + '.jpg')
            file_count += 1
        count += 1

def _main():
    if args.subcommand == 'images':
        for image_file_name in os.listdir(test_images_path):
            try:
                image_type = imghdr.what(os.path.join(test_images_path, image_file_name))
                if not image_type:
                    continue
            except IsADirectoryError:
                continue

            image = Image.open(os.path.join(test_images_path, image_file_name))
            test_yolo(image, image_file_name)


    if args.subcommand == 'npz':
        npz_obj = np.load(test_npz_path)
        images = npz_obj['images']

        for image_index, image_arr in enumerate(images):
            image = Image.fromarray(image_arr)
            test_yolo(image, str(image_index) + '.jpg')

    if args.subcommand == 'mp4':
        process_mp4(test_mp4_vod_path)

    if args.subcommand == 'youtube':
        youtube_download_path = os.path.dirname(os.path.abspath(__file__))
        youtube_download_path = os.path.join(youtube_download_path, "output")
        if os.path.exists(youtube_download_path + '/vod.mp4'):
            os.remove(youtube_download_path + '/vod.mp4')
        ydl_opts = {'outtmpl': youtube_download_path + '/' + 'vod_full.%(ext)s', 'format': '137'}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([test_youtube_link])
        # TODO test this on other systems.
        print("Calling ffmpeg to cut up video")
        call(['ffmpeg', '-i', youtube_download_path + '/vod_full.mp4', '-ss', start_time, '-to', end_time, '-c', 'copy', youtube_download_path + '/vod.mp4'])
        os.remove(youtube_download_path + '/vod_full.mp4')
        print("Done with ffmpeg")
        process_mp4(youtube_download_path + '/vod.mp4')

    sess.close()

if __name__ == '__main__':
    _main()
