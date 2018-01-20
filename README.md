# DeepLeague + Dataset of over 100,000 labeled images to further computer vision research within eSports - by Farza  

![Alt text](https://media.giphy.com/media/3ohc0PVVsgt578uBkA/giphy.gif)
### Disclaimer
I wrote the majority of this code in 5 days during a hurricane when I was bored. My code isn't horrible but it defnitely isn't a nice and fancy library with amazing documentation. There are likely many bugs and inefficiencies.

BUT, getting setup to run the test script is easy enough and I'll help you out there but if you want to actually mess with the core code you'll be mostly on your own.

If you have questions contact me on [Twitter](https://twitter.com/FarzaTV).

### How do I get DeepLeague?

You'll need [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), [conda](https://conda.io/docs/user-guide/install/index.html), and [brew](https://brew.sh/). Once you install them you can check if everything works okay by typing in these commands in your terminal. I've confirmed that these steps work on Mac OS and Ubuntu. Windows Users, you're on your own :(. But it shouldn't be to tough if you know your way around the command line.

```sh
$ conda
$ git
$ brew
```

If you were able to run those three commands without any errors, you can continue.
```sh
$ git clone https://github.com/farzaa/DeepLeague.git
$ cd DeepLeague
$ cd YAD2K
$ conda create --n DeepLeague python=3.6
$ source activate DeepLeague
$ conda install python.app # this install python as a framework for mat plot lib.

$ pip install opencv-python youtube_dl
$ conda install -c menpo ffmpeg
$ pip install numpy h5py pillow matplotlib
$ pip install tensorflow
$ pip install keras

$ brew install wget
$ brew install ffmpeg --with--libvpx
$ wget http://pjreddie.com/media/files/yolo.weights
$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
$ python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
```
Running that last command is extremely important. It might produce some errors which you can hopefully Google and quickly solve. I've found it really is dependent on your system + hardware.

You are almost good to go. Last thing you need is get the [file for the weights](https://drive.google.com/open?id=1-r_4Ex3OC-MTcTwNE7xJkdpiSz_3rb8A). This is the core file behind the magic of DeepLeague

Download this and put it in the /YAD2K directory. The test script will expect it to be here.


### How do I run DeepLeague?
Honestly, this repo has so many tiny fucntions. But, let me explain the easiest way to get this going if all you want to do is analyze a VOD (which most of you want I presume). the ```test_deep_league.py``` is the key to running everything. Its a little command line tool I made that lets you input a VOD to analyze using three different ways: a YouTube link, path to local MP4, path to a directory of images. I like the YouTube link option best, but if you have trouble with it feel free to use the MP4 approach instead. All you need is a 1080P VOD of a League game. Its extremely important its 1080p or else my scripts will incorrectly crop the mini map. Also, DeepLeague is only trained on mini maps from 1080P video. Other sizes aren't tested.

Here's an example of me running the tool with a YouTube link. This method automatically downloads the YT video as well and cuts it up according to the the start and end time you gave it. It will automatically do all the renaming to process stuff.

This command specifies to start at the 30 second mark and end 1 minute in. This is useful when you only want to analyze a part of a VOD. The frames that are output are saved to the "output" folder as specified by the command below.

```sh
python test_deep_league.py -out output youtube -yt https://www.youtube.com/watch?v=vPwZW1FvtWA -yt_path /output -start 0:00:30 -end 0:01:00
```

You should first see the download start:

![Alt Text](https://media.giphy.com/media/l49JQHcc04ZyYX3t6/giphy.gif)

Then you should see DeepLeague start predicting bounding boxes.

![Alt text](https://media.giphy.com/media/3oFzlYZnMiO1wSsc0g/giphy.gif)

If you want to use a local mp4 file that you recorded yourself use the command below where -mp4 tells the script where the VOD is on your computer.

```sh
python test_deep_league.py -out output mp4 -mp4 /Volumes/DATA/data/data/C9_CLG_G_2_MARCH_12_2017/vod.mp4
```

### How do I get the dataset:

I've split the dataset into multiple .npz files so it isn't just one massive file. I mainly did this to make batch training easier. Plus, its really annoying when you are downloading one big file and that download randomly fails and you need to start all over.

Also, I have already split the dataset into training, testing, and validation sets which splits the data into 80%, 17.5%, and 2.5% cuts respectively.

These .npz files only have the cropped mini maps frames and the bounding box information associated with every frame. If that's all you want, perfect. You can download it here.

If you want help reading this npz file, check out ```def visualize_npz_data``` [here](https://github.com/farzaa/DeepLeague/blob/master/vis_data.py).
