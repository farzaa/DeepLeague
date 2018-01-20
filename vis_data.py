import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from read_ocr_and_lolesport_data import get_game_data_dict
from paths import BASE_DATA_PATH
import argparse
import colorsys
import random
import numpy as np
import os
import re
import time

parser = argparse.ArgumentParser(description='Visualize npz cluster ground truth data (or game_data object)')
parser.add_argument('--path', type=str, default='data_training_set_cluster_0.npz', help='path to npz cluster to visualize')
# parser.add_argument('--classes', type=str, default='YAD2K/model_data/league_classes.txt', help='path to .txt file that holds the classes')
# parser.add_argument('--dataset_type', type=str, default='npz', help='npz or game object')
# parser.add_argument('--refresh_rate', type=int, default=1000, help='# of seconds between showing each new frame')

# credit for box drawing utils to YAD2K guys.
def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image.

    Draw bounding boxes with class name and optional box score on image.

    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    image = Image.fromarray(np.floor(image).astype('uint8'))
    font_path = 'YAD2K/font/FiraMono-Medium.otf'
    font = ImageFont.truetype(
        font= font_path,
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]

        if isinstance(scores, np.ndarray):
            score = scores[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

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
                [left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)

# TODO: make this function play nice with draw_boxes
def visualize_game_data(game_data, folder):
    for time_stamp in game_data:
        frame_obj = game_data[time_stamp]
        # not all frame objs have actual frames asscoiated with them!
        if frame_obj.frame_path is not None:
            print("%s .... %s" % (frame_obj.time_obj.time_as_string, frame_obj.frame_path))

            x_val = int(frame_obj.game_snap['playerStats']['1']['x'])
            y_val = int(frame_obj.game_snap['playerStats']['1']['y'])
            print("X... ", x_val)
            print("Y.... ", y_val)

            x_val_2 = int(frame_obj.game_snap['playerStats']['2']['x'])
            y_val_2 = int(frame_obj.game_snap['playerStats']['2']['y'])

            print("X2... ", x_val_2)
            print("Y2.... ", y_val_2)

            # load in image w/ PIL for easy drawing / cropping
            im = Image.open(BASE_DATA_PATH + folder  + '/frames/' + frame_obj.frame_path).crop((1625, 785, 1920, 1080))
            draw = ImageDraw.Draw(im)
            draw.rectangle([(x_val - 19, y_val - 20),(x_val + 11, y_val + 10)], outline='red')
            draw.rectangle([(x_val_2 - 19, y_val_2 - 20),(x_val_2 + 11, y_val_2 + 10)], outline='blue')

            print('\n')
            # now draw using PIL
            del draw

            im = np.array(im, dtype = np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB )
            cv2.imshow("IMAGE", im)
            if cv2.waitKey(1) == ord('q'):
                break

# to visualize dataset after all processing
def visualize_npz_data(npz_file_path, all_classes, refresh_rate):
    np_obj = np.load(npz_file_path)

    for image, boxes in zip(np_obj['images'], np_obj['boxes']):
        print(boxes)
        img = Image.fromarray(image)

        box_to_draw = []
        for box in boxes:
            box_to_draw.append([box[2], box[1], box[4], box[3]])

        print(boxes)
        img = np.array(img, dtype = np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw_boxes(img, box_to_draw, [boxes[i][0] for i in range(len(boxes))], all_classes)

        cv2.imshow("IMAGE", img)
        if cv2.waitKey(refresh_rate)  ==  ord('q'):
            break
def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def visualize_frames_on_drive(images_path, refresh_rate):
    images = os.listdir(images_path)
    images = sorted_nicely(images)
    first_run = True
    for image_file in images:
        print(image_file)
        im = cv2.imread(images_path + image_file)
        im2 = cv2.imread("/Users/flynn/Documents/DeepLeague/YAD2K/frames_no_labels/" + image_file)
        cv2.imshow("IMAGE", im)
        cv2.imshow("IMAGE2", im2)
        if cv2.waitKey(refresh_rate)  ==  ord('q'):
            break
        if first_run:
            first_run = False
            time.sleep(15)


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path
    # classes_path = args.classes
    # dataset_type = args.dataset_type
    # refresh_rate = args.refresh_rate

    # all_classes = get_classes(classes_path)
    visualize_frames_on_drive(path, 80)
    # at this point, game_data has a dict of frame objects with a frame_path IF one was found for it.
    # game data's keys are in order of game time.
    # game_data = get_game_data_dict('GAM_LZ_G_1_OCTOBER_6_2017')
    # visualize_game_data(game_data, 'GAM_LZ_G_1_OCTOBER_6_2017')

    # TODO: allow for visualization of game_data as well!
    if dataset_type == 'npz':
        visualize_npz_data(path, all_classes, refresh_rate)
