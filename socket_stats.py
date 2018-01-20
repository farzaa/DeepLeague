# this script just prints some of the socket data in a readable way
import os
import json
import operator
from paths import BASE_DATA_PATH


# keeps ref to all the frames we want.
folders_to_save = []
debug = False
# goes through every json and figures out what champs were involved in the game.
def check_champs(json_file, champ_dict, folder_name):
    # ROUGH # of frames. Doesn't account for when champ is dead!
    frame_count = len(os.listdir(BASE_DATA_PATH + folder_name + '/frames'))
    # first check if we have enough of champ.
    stats = json_file[0]['playerStats']
    frame_list = []
    for i in range(1, 11):
        champ = stats[str(i)]['championName']
        if champ in champ_dict:
            frame_list.append(champ_dict[champ]['frame_count'])
    # all champs already have 20k + frames, thats enough!
    print(frame_list)

    # we only want to throw away the game if all 10 champs are included and one of them have frames > 100000
    if len(frame_list) != 0 and len(frame_list) == 10 and any(i >= 50000 for i in frame_list):
        return
    # throw away game if we already have 20000+ frames for all 10 champs in this game.
    # this is equivalent to around 10 games of data.
    if len(frame_list) != 0 and len(frame_list) == 10 and all(i >= 20000 for i in frame_list):
        return

    for i in range(1, 11):
        champ = stats[str(i)]['championName']
        if champ not in champ_dict:
            champ_dict[champ] = {'game_count': 0, 'frame_count': 0}

        if champ in champ_dict:
            # increment game count
            champ_dict[champ]['game_count'] += 1
            # increment frame count
            champ_dict[champ]['frame_count'] += frame_count

    # these will be all the frames we want.
    folders_to_save.append(folder_name)


def read_json():
    champ_dict = {}
    i= 0
    ocr = 0
    for folder_name in os.listdir(BASE_DATA_PATH):

        print("Opening ", folder_name)
        if os.path.isdir(BASE_DATA_PATH + folder_name):
            if os.path.exists(BASE_DATA_PATH + folder_name + '/frames'):
                if not os.path.exists(BASE_DATA_PATH + folder_name + '/time_stamp_data_clean.json'):
                    ocr += len(os.listdir(BASE_DATA_PATH + folder_name + '/frames'))
                    print("Still need ocr data for %s and OCR is count is @ %d" % (folder_name, ocr))
                    continue
            else:
                print("FRAMES doesn't exist for ", folder_name)

            try:
                json_file = json.load(open(BASE_DATA_PATH + folder_name + '/socket.json', 'r'))
            except Exception as e:
                print("JSON failed on ", folder_name)
                print(e)
                continue

            check_champs(json_file, champ_dict, folder_name)
            print(champ_dict)
            i+=1
            if debug and i == 2:
                break

    sorted_keys = sorted(champ_dict.items(), key=lambda x: x[1]['frame_count'], reverse=True)
    print(sorted_keys)
    for key in sorted_keys:
        print(key)

    return sorted_keys

def get_me_folders_and_label_dict():
    full_champs_dict = read_json()
    # create a label dict for YOLO
    label_dict_yolo = {}
    champ_number = 0
    for tup in full_champs_dict:
        # we have enough frames to train the champ!
        if tup[1]['frame_count'] > 20000:
            label_dict_yolo[tup[0]] = champ_number
            champ_number += 1

    print(folders_to_save)
    print(label_dict_yolo)

    return folders_to_save, label_dict_yolo

if __name__ == '__main__':
    get_me_folders_and_label_dict()
