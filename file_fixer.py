# i want image files arranged numerically when in a listdir
# i use a simple lamda fucntion for this
import os

def sort_files_numerically(full_path_to_files):
    files = os.listdir(path_to_files)

    # only care about jpg frames.
    for file in files:
        if(file.split(".")[1] != "jpg"):
            files.remove(file)

    # trick to sort files in numerical order given format "frame_#.jpg"
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[1].split(".")[0]))
