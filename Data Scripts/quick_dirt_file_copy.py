import shutil
import os
import sys
import time

for folder in os.listdir('/Users/flynn/Documents/DeepLeague/data'):
    print('looking @ ', folder)
    if os.path.isdir('/Users/flynn/Documents/DeepLeague/data/' + folder):
        if not (os.path.exists('/Volumes/DATA/data/' + folder)):
            os.mkdir('/Volumes/DATA/data/' + folder)
        for file_name in os.listdir('/Users/flynn/Documents/DeepLeague/data/' + folder):
            # frames folder special case
            if 'frames' in file_name:
                print("creating / copying frames")
                os.mkdir('/Volumes/DATA/data/' + folder + '/frames/')
                time.sleep(3000)
                src = '/Users/flynn/Documents/DeepLeague/data/' + folder + '/frames/'
                dst = '/Volumes/DATA/data/' + folder + '/frames/'
                shutil.copytree(src, dst)

            print('looking @ file ', file_name)
            src = '/Users/flynn/Documents/DeepLeague/data/' + folder + '/' + file_name
            dst = '/Volumes/DATA/data/' + folder
            shutil.copy(src, dst)
