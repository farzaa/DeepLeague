import youtube_dl
from subprocess import call
import shutil
import os
import sys
import csv

# tuples of file_name, youtube_link, game_start_time, game_end_time
CSV_FILE_PATH = '/Users/flynn/Documents/DeepLeague/data/vod_info.csv'


with open(CSV_FILE_PATH, 'rt') as csvfile:
    reader = csv.reader(csvfile)

    first_run = True
    check_me = []
    for row in reader:
        # skip CSV header
        if first_run:
            first_run = False
            continue

        folder_name = row[0]
        link = row[1]
        start_time = row[2]
        end_time = row[3]
        json_file = row[4]

        # print("Link ", link)
        print("Folder ", folder_name)

        check_me.append(folder_name)
        if not os.path.exists('data/' + folder_name):
            print("Creating output path for ", folder_name)
            os.mkdir('data/' + folder_name)

        # if we already have the video move on!
        if os.path.exists('data/' + folder_name + '/vod.mp4'):
            continue

        if os.path.exists('new_games/games/' + json_file + '.json'):
            if not os.path.exists('data/' + folder_name + '/socket.json'):
                print("JSON File exists for %s in the completed_games folder", folder_name)
                shutil.copy('new_games/games/' + json_file + '.json','data/' + folder_name + '/socket.json')
                print('Copied to folder in "data" directory!')

        else:
            print("JSON file %s for %s didn't exist!" % (json_file, folder_name))
            sys.exit(1)

        ydl_opts = {'outtmpl': 'data/' + folder_name + '/' + 'vod_full.%(ext)s', 'format': '137'}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
            info_dict = ydl.extract_info(link, download=False)
            video_title = info_dict.get('title', None)

        print("Calling ffmpeg")
        call(['ffmpeg', '-i', 'data/' +  folder_name + '/vod_full.mp4', '-ss', start_time, '-to', end_time, '-c', 'copy', 'data/' + folder_name + '/vod.mp4'])
        os.remove('data/' +  folder_name + '/vod_full.mp4')
        print("Done with ffmpeg")
if len(check_me) != len(set(check_me)):
    print("NOT EQUAL")

for item in set(check_me):
    check_me.remove(item)
print(check_me)
