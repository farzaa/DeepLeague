<<<<<<< HEAD
# DeepLeague - by Farza  

Yo whats up boys n girls my name's **Farza** and I'm an undergrad CS student at UCF. Hope you find my little creation useful! To read more about how I actually built this thing you can read the blog post here. And be sure to follow me on Twitter [here](https://twitter.com/farzatv?lang=en). I'm semi-lit. Also, very sorry if the code is ugly! I wrote most of it in two days.

### So what is it?

Note: DeepLeague is built upon some complicated concepts. But, don't get discouraged! I made this so that anyone with a little knowledge about code and the command line would be able to use it!

DeepLeague is the first program of its kind that combines computer vision, deep learning, and League of Legends in order to move LoL analytics to the next level by giving developers **easy access** to the data encoded in the *pixels* of the game. Don't worry, I did all the heavy lifitng! It allows users to *input* just a League of Legends VOD and will *output* the coordinate location of all the champions in the game at every frame by **"watching"** the minimap. So, imagine you gave DeepLeague a VOD. It would output this:

I also offer a dataset of over 400,000 minimap images *labeled* with bounding boxes/ classes for you to train your own neural network. DeepLeague can do so much more, and you can help! You can read more about this in the Dataset section.

### Wait wtf is the point?

Riot API only allows people to get *post-game data*, things like kills, deaths, items purchased, etc. In terms of analytics, theres only so much you can do with this boring data. But what if you could get *live*  data during the game while a game is actually happening? Imagine if you called an API and got back information about stuff actually happening during that live game, things like: positions of the champions at specific timestamps, location of wards, status of towers, what items a person bought, when a champion is dead, etc. That'd be super cool! Just imagine all the insane analytics we'd be able to run on that. But, Riot simply does not allow it for obvious security reasons.

BUT, this is all data you can get by actually watching the game, right? Though, we don't want to watch thounsands of hours of League of Legends! And this is where DeepLeague comes in.

DeepLeague allows users to input a League of Legends VOD and will predict the location of all the champions in the game at every frame by **"watching"** the minimap. It does this by running the VOD through a deep convolutional neural net that I trained (on a dataset I gathered) which specializes in predicitng bounding boxes + classes. To read more about how things work, you can read the blog post here.

### Cool, what can I do with it?
The possibilties with DeepLeague are endless and I really want people to create cool stuff based off it!

Here's a couple things you could do given the output of DeepLeague.
- analyze how the jungler paths, where he starts his route, when he ganks, when he backs, which lane he exerts the most pressure on, when mid roams, which lanes mid ganks when it roams
- analyze when laners overextend, when laners die, when they are getting pressured by other lanes, when they are losing lane, when they die, when they leave laning phase, when they get a solo kill
- analyze when teams set up dives, when they decide to do dragon, how they rotate around the map as a team, how they set up baron, when they teamfight, when they hide in brushes

### Enough talk! How do I get it?
Don't worry you won't need a Ph.D in AI to get this running! (good joke Farza hahahahaha kms!)

All you need to get going in is [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [conda](https://conda.io/docs/user-guide/install/index.html) for Python 3.6. Once you install them you can check if everything works okay by typing in these commands in your terminal.
```sh
$ conda
$ git
```
If you were able to run those three commands without any errors, you can continue.
```sh
$ git clone --recursive https://github.com/farzaa/DeepLeague.git
$ cd DeepLeague
$ cd YAD2K
$ conda create -n DeepLeague python=3.6
$ source activate DeepLeague


TODO: switch yolo weights with deep league weights
pip install opencv-python
pip install youtube_dl
conda install -c menpo ffmpeg

$ wget http://pjreddie.com/media/files/yolo.weights
$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
$ ./yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
```
At this point you are good to go. To run DeepLeague, just do:


### Dataset
Right now, DeepLeague only tracks about 50 champions because that's how I trained it. But it can be so much more and could support things like more champions, wards, towers, dragon, baron, and more. And don't worry, I give you the data and the code to train your neural net. All you have to do is get creative with the data! Let me talk more about how things are formatted.

First lets get the dataset:


So this dataset has a lot of stuff. I have over 260 VODs. Each VOD is broken up into a bunch of .jpg's where we simply scrape the VOD at 1 frame per second. Each VOD is associated with a file called time_stamp_data_clean.json which tells you the *in-game* time stamp of the frames associated with that VOD. This is the value of the in-game clock. You'll see why this usefull in the next paragraph.

TODO: Scripts to get data.

With each VOD there is a massive json file associated with it that has information associated with **every timestamp** in the game. This json is called socket.json. socket.json has a lot of stuff. For example, things like champion coordinates, health, mana, wards placed, and much more. You can download an example of one here. The most important thing to understand with this is that it gives data for every single second of the game so you can get very creative with how you make your dataset. You can use this along with time_stamp_data_clean.json and the frames to get your neural net going.


### Todos

 - Write MORE Tests
 - Add Night Mode
