# calls Google Vision API to do OCR, and saved results to a CSV.
# i use the API vs Tesseract for performance reasons.
# credit for orignal program i altered - https://gist.github.com/dannguyen/a0b69c84ebc00c54c94d



import base64
from io import BytesIO
from os import makedirs
from os.path import join, basename
import json
import requests
import os
from PIL import Image
from pprint import pprint
import re
import sys
from read_ocr_and_lolesport_data import convert_string_time_to_easy_time
from paths import BASE_DATA_PATH

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
RESULTS_DIR = 'jsons'
makedirs(RESULTS_DIR, exist_ok=True)


API_KEY = "AIzaSyDZKJ00w4fQX-Z9xxGIY5pykptFv3V7YGI"



ocr_counter = 0
def make_image_data_list(folder, image_filenames):
    """
    image_filenames is a list of filename strings
    Returns a list of dicts formatted as the Vision API
        needs them to be
    """
    img_requests = []
    for imgname in image_filenames:
        # load as PIL
        img = Image.open(BASE_DATA_PATH + folder + '/frames/' + imgname).crop((900, 75, 1000, 125))
        buffer_str = BytesIO()
        img.save(buffer_str, 'jpeg')
        # convert to base64 for api
        encoded_string = base64.b64encode(buffer_str.getvalue()).decode()
        img_requests.append({
                'image': {'content': encoded_string},
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 1
                }]
        })
    return img_requests

def make_image_data(folder, image_filenames):
    """Returns the image data lists as bytes"""
    imgdict = make_image_data_list(folder, image_filenames)
    return json.dumps({"requests": imgdict }).encode()


def request_ocr(api_key, folder, image_filenames):
    response = requests.post(ENDPOINT_URL,
                             data=make_image_data(folder, image_filenames),
                             params={'key': api_key},
                             headers={'Content-Type': 'application/json'})
    return response


def create_data_json(folder):
    # get file paths
    image_paths = os.listdir(BASE_DATA_PATH + folder + '/frames/')

    # regex for times (see later below)
    reg = re.compile('^[1-5]?[0-9]:[0-9][0-9]$')

    # only care about jpg frames.
    for file in image_paths:
        if(file.split(".")[1] != "jpg"):
            image_paths.remove(file)

    # sort in numerical order
    sorted_files = sorted(image_paths, key=lambda x: int(x.split('_')[1].split(".")[0]))

    # file we will write all data too
    outfile = open(BASE_DATA_PATH + folder +  '/time_stamp_data_dirty.json', 'w')
    print("Created game_data json (dirty) at folder ", folder)
    outfile.write('[\n')
    json.dump({'info': folder}, outfile)
    outfile.write(',\n')

    counter = 0
    for i in range(0, len(sorted_files) - 15, 15):
        counter = i

    # we only want to process 15 images per request to keep things organized
    for i in range(0, len(sorted_files) - 15, 15):
        image_paths = []
        if i % 100 == 0:
            print("On frame %d out of %d" % (i, len(sorted_files) - 15))

        for j in range(0, 15):
            # just create a list of 15 files
            image_paths.append(sorted_files[i + j])
        response = request_ocr(API_KEY, folder, image_paths)

        if response.status_code != 200 or response.json().get('error'):
            print(response.text)
        else:
            global ocr_counter
            ocr_counter += 15
            if ocr_counter % 100:
                print("OCR counter is currently at ", ocr_counter)
            if ocr_counter > 250000:
                sys.exit(1)
            # print(len(response.json()['responses']))
            # print(response.json()['responses'][-1])
            # print(i)
            # print(counter)

            # TODO Really annoying problem here. Basically I need to be able to handle two cases when appending the final comma to the json
            # 1. when i reach the END (this ones ez)
            # 2. when i reach the last response, and the last item in the response is BAD. (ex. fails on regex)
            for idx, resp in enumerate(response.json()['responses']):
                if "textAnnotations" not in resp:
                    continue

                t = resp['textAnnotations'][0]
                time = t['description'].strip()

                imgname = image_paths[idx]

                # check if time is in the proper format
                # lots of times you get things like pauses, or replays, and we don't want those frames.
                if reg.match(time) is None:
                    continue

                # create json objct to save data
                obj = {'file_name': imgname, 'time': time}
                # save to JSON file
                json.dump(obj, outfile)

                # TODO make this prettier
                if idx == len(response.json()['responses']) - 1 and counter - 14 <= i <= counter:
                    continue
                else:
                    outfile.write(',\n')

    outfile.write(']\n')
    outfile.close()

def create_clean_data_json(folder):
    try:
        frame_timestamp_data = json.load(open(BASE_DATA_PATH + folder + '/time_stamp_data_dirty.json', 'r'))
    except Exception as e:
        print("JSON FAILED FOR %s CLEAN NEVER CREATED. FIX JSON." % folder)
        return
    outfile = open(BASE_DATA_PATH + folder + '/time_stamp_data_clean.json', 'w')
    print("Created game_data json (clean) at folder ", folder)
    first = True
    prev_minutes = None
    outfile.write('[\n')
    json.dump({'info': folder}, outfile)
    outfile.write(',\n')

    for i, time_frame in enumerate(frame_timestamp_data):
        # skip over first item in json
        if first:
            first = False
            continue
        time_obj = convert_string_time_to_easy_time(time_frame['time'])
        curr_minutes = time_obj.minutes
        if prev_minutes is not None:
            # check if the time difference is greater than 5 min from prev frame.
            # if so, its most likely an incorrect time returned from OCR.
            if abs(curr_minutes - prev_minutes) > 5:
                continue
        prev_minutes = curr_minutes
        json.dump(time_frame, outfile)

        if i == len(frame_timestamp_data) - 1:
            continue
        else:
            outfile.write(',\n')

    outfile.write(']\n')
    outfile.close()

if __name__ == '__main__':
    for folder_name in os.listdir(BASE_DATA_PATH):
        print("Currently on... ", folder_name)
        if os.path.isdir(BASE_DATA_PATH + folder_name):
        #     items =  os.listdir(BASE_DATA_PATH + folder_name)
        #     if ('time_stamp_data' not in str(items)):
        #         print("No time_stamp_data file in folder ", folder_name)
        #         folders.append(folder_name)
            # creates a "dirty" json file
            if not os.path.exists(BASE_DATA_PATH + folder_name + '/time_stamp_data_dirty.json'):
                create_data_json(folder_name)

            # creates a "clean" json file where i remove some time stamps that don't make sense
            # i create two versions in case i find a bug later with my "clean" code
            if not os.path.exists(BASE_DATA_PATH + folder_name + '/time_stamp_data_clean.json'):
                create_clean_data_json(folder_name)
