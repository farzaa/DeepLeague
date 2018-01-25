# shouts out to this guy for helping me out!
# https://github.com/shadySource/DATA/blob/master/package_dataset.py

from read_ocr_and_lolesport_data import get_game_data_dict
import numpy as np
from PIL import Image
from random import shuffle
import os
from paths import  BASE_DATA_PATH
from socket_stats import get_me_folders_and_label_dict
import sys

label_dict = {'Lulu': 0, 'Ezreal': 1, 'Rengar': 2, 'Orianna': 3, 'Karma': 4, 'Nautilus': 5, 'Syndra': 6, 'Gragas': 7, 'Elise': 8, 'Ashe': 9, 'Shen': 10, 'LeeSin': 11, 'Graves': 12, 'Caitlyn': 13, 'Malzahar': 14, 'Rumble': 15, 'Jhin': 16, 'Khazix': 17, 'Renekton': 18, 'Thresh': 19, 'Jayce': 20, 'Varus': 21, 'Maokai': 22, 'Taliyah': 23, 'Zyra': 24, 'Sejuani': 25, 'Lucian': 26, 'Tristana': 27, 'JarvanIV': 28, 'Vladimir': 29, 'Nami': 30, 'Janna': 31, 'TahmKench': 32, 'Ahri': 33, 'Xayah': 34, 'Cassiopeia': 35, 'Viktor': 36, 'Rakan': 37, 'Galio': 38, 'Camille': 39, 'Poppy': 40, 'Olaf': 41, 'Ryze': 42, 'KogMaw': 43, 'RekSai': 44, 'Leblanc': 45, 'Kled': 46, 'Ekko': 47, 'Talon': 48, 'Fizz': 49, 'Morgana': 50, 'Sivir': 51, 'Twitch': 52, 'Chogath': 53, 'Ziggs': 54, 'Kennen': 55}


debug = False
shuffle = True


# returns array with label, x_min, y_min, x_max, y_max for a single champ
def get_box_for_champ(champ_index, frame_obj):
    champ_name = frame_obj.game_snap['playerStats'][champ_index]['championName']
    label = label_dict[champ_name]
    x_val = int(frame_obj.game_snap['playerStats'][champ_index]['x'])
    y_val = int(frame_obj.game_snap['playerStats'][champ_index]['y'])

    # label xmin ymin xmax ymax
    return np.array([label, x_val - 19, y_val - 20, x_val + 11, y_val + 10])

# a champ is dead when its x/y is 0. we want to throw these away.
def dead(frame_obj, champ_index):
    if  frame_obj.game_snap['playerStats'][champ_index]['x'] == 0 and frame_obj.game_snap['playerStats'][champ_index]['y'] == 0:
        return True
    return False

def check_boxes_for_champs_in_dict(frame_obj):
    boxes_in_frame = []
    for i in range(1, 11):
        champ_index = str(i)
        champ_name = frame_obj.game_snap['playerStats'][champ_index]['championName']

        # for now only care about specific champs in our labels. not eveything.
        # our neural net only cares about data we have labels for
        if champ_name not in label_dict:
            continue

        # if champ is dead, we dont care for their position
        if dead(frame_obj, champ_index):
            continue

        # this will always return a box, no matter what.
        # every champ always has a position
        box = get_box_for_champ(champ_index, frame_obj)
        # we want to remove the possibility of negative coordinates after comnversion of coordinates
        if(box[1] < 0 or box[2] < 0 or box[3] < 0 or box[4] < 0):
            continue

        boxes_in_frame.append(box)
    if len(boxes_in_frame) == 0:
        return boxes_in_frame, True
    return boxes_in_frame, False

def get_bounding_boxes_and_images(game_data, folder):
    all_boxes = []
    all_images = []

    counter = 0
    for time_stamp in game_data:
        boxes_in_timestamp = []
        frame_obj = game_data[time_stamp]
        if frame_obj.frame_path is not None:

            # i found that data before the 3 min mark was usually incomplete and had champs not labeled. this is bad!
            # i do this here rather than in create_data() because i don't wanna augment the original data.
            if frame_obj.time_obj.minutes < 3:
                continue

            # go through every champion
            boxes, empty = check_boxes_for_champs_in_dict(frame_obj)
            # if the frame came back with no boxes, we don't care for it. it has no data we care about.
            if empty:
                continue

            all_boxes.append(np.array(boxes))

            # image work
            im = Image.open(BASE_DATA_PATH + folder + '/frames/' + frame_obj.frame_path).crop((1625, 785, 1920, 1080))
            im = np.array(im, dtype = np.uint8)
            all_images.append(im)

            if debug:
                print(counter)
                if counter == 10:
                    break
            counter += 1

            if len(all_images) % 50 == 0:
                print("Image array length... ", len(all_images))
    print(all_boxes)
    return all_boxes, all_images

# takes batch of games, shuffles frames and saves to train, val, and test clusters
def create_cluster_from_folders(folder_list, iterator):
    all_boxes = []
    all_images = []
    for folder_name in folder_list:
        if os.path.isdir(BASE_DATA_PATH + folder_name):
            print("Aggregating data for ", folder_name)
            game_data = get_game_data_dict(folder_name)
            if game_data is None:
                continue
            # arrange all data in to these two nice lists
            boxes, images = get_bounding_boxes_and_images(game_data, folder_name)
            print("Appending %d images " % len(images))
            all_boxes.extend(boxes)
            all_images.extend(images)

            print("Length of boxes arr ", len(all_boxes))

    if shuffle:
        np.random.seed(13)
        indices = np.arange(len(all_images))
        all_images = np.asarray(all_images)
        all_boxes = np.asarray(all_boxes)
        np.random.shuffle(indices)
        all_images, all_boxes = all_images[indices], all_boxes[indices]


    # save 2.5% of frames per cluster for testing.
    test_set_cut = int(0.025 * len(all_images))
    # 17.5 % val cut per cluster.
    val_set_cut = int(0.175 * len(all_images))
    train_set_cut = int(0.80 * len(all_images))

    print(all_images)
    all_images = np.array(all_images)
    all_boxes = np.array(all_boxes)


    train_images, val_images, test_images = np.split(all_images, [train_set_cut, train_set_cut + val_set_cut])
    train_boxes, val_boxes, test_boxes = np.split(all_boxes, [train_set_cut, train_set_cut + val_set_cut])

    print("Shape of training images... ", train_images.shape)
    print("Shape of val images... ", val_images.shape)
    print("Shape of test images... ", test_images.shape)

    print("Shape of training boxes... ", train_boxes.shape)
    print("Shape of val boxes... ", val_boxes.shape)
    print("Shape of test boxes... ", test_boxes.shape)

    # we want to hand all these off as a list!
    # train_images, val_images, test_images = train_images.tolist(), val_images.tolist(), test_images.tolist()
    # train_boxes, val_boxes, test_boxes = train_boxes.tolist(), val_boxes.tolist(), test_boxes.tolist()

    np.savez("/Volumes/DATA/clusters_cleaned/train/data_training_set_cluster_" + str(iterator), images=train_images, boxes=train_boxes)
    np.savez("/Volumes/DATA/clusters_cleaned/test/data_test_set_cluster_" + str(iterator), images=test_images, boxes=test_boxes)
    np.savez("/Volumes/DATA/clusters_cleaned/val/data_val_set_cluster_" + str(iterator), images=val_images, boxes=val_boxes)

    print("Saved @ # " + str(iterator))



if __name__ == '__main__':
    folder_list, label_dict_yolo = get_me_folders_and_label_dict()
    chunks_of_folders = [folder_list[i:i + 10] for i in range(0, len(folder_list), 10)]

    iterator = 0
    for chunk in chunks_of_folders:
        create_cluster_from_folders(chunk, iterator)
        iterator += 1
