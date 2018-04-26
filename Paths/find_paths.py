import cv2
import os
import sys

minimap_crop = (805, 1080, 1645, 1920)

# red side crops (left to right)
red_krugs_crop = (35, 65, 105, 135)
red_red_buff_crop = (55, 95, 110, 150)
red_chickens_crop = (85, 115, 125, 155)
red_wolves_crop = (105, 135, 185, 215)
red_toad_crop = (135, 165, 210, 240)
red_blue_buff_crop = (125, 165, 180, 220)

# blue side crops (left to right)
blue_toad_crop = (105, 135, 30, 60)
blue_blue_buff_crop = (105, 145, 55, 95)
blue_wolves_crop = (135, 165, 55, 85)
blue_chickens_crop = (155, 185, 115, 145)
blue_red_buff_crop = (170, 210, 120, 160)
blue_krugs_crop = (205, 235, 140, 170)

# scuttle crab crops
bot_side_scuttle_crop = (160, 190, 175, 205)
top_side_scuttle_crop = (85, 115, 70, 100)


sample_image_path = 'frames/372.jpg'


display_image = None

def sort_files_numerically(full_path_to_files):
    files = os.listdir(full_path_to_files)

    # only care about jpg frames.
    for file in files:
        if(file.split(".")[1] != "jpg"):
            files.remove(file)

    # trick to sort files in numerical order given format "frame_#.jpg"
    sorted_files = sorted(files, key=lambda x: int(x.split(".")[0]))
    return sorted_files


def do_template_match(crop, mini_map_img, template):
    img = mini_map_img.copy()
    img = img[crop[0]:crop[1], crop[2]:crop[3]]

    h, w, c = template.shape
    print(w, h, c)
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val, max_val, min_loc, max_loc)

    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    print((crop[0], crop[2]), (crop[1], crop[3]))

    if min_val < 0.05:
        cv2.rectangle(display_image, (crop[2], crop[0]), (crop[3], crop[1]), 255, 2)
        return True
    else:
        return False

# returns a dictionary indicating which jungle camps are "seen" by the template match.
def camps_seen_for_frame(path_to_frame):
    camps_seen = {
        'red_toad': None,
        'red_blue_buff': None,
        'red_wolves': None,
        'red_chickens': None,
        'red_red_buff': None,
        'red_krugs': None,
        'blue_toad': None,
        'blue_blue_buff': None,
        'blue_wolves': None,
        'blue_chickens': None,
        'blue_red_buff': None,
        'blue_krugs': None,
        'top_side_scuttle': None,
        'bot_side_scuttle': None
    }

    display_image = cv2.imread(path_to_frame, cv2.IMREAD_COLOR)[minimap_crop[0]:minimap_crop[1], minimap_crop[2]:minimap_crop[3]]

    # assuming a full 1920x1080 image coming in.
    mini_map_img = display_image.copy()

    cv2.imwrite('minimap_example.jpg', cv2.imread(sample_image_path, cv2.IMREAD_COLOR)[minimap_crop[0]:minimap_crop[1], minimap_crop[2]:minimap_crop[3]])

    for template_name in os.listdir('templates/'):
        if 'red_toad_template' in template_name:
            template = cv2.imread('templates/red_toad_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(red_toad_crop, mini_map_img, template)
        if 'red_blue_buff_template' in template_name:
            template = cv2.imread('templates/red_blue_buff_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(red_blue_buff_crop, mini_map_img, template)
        if 'red_wolves_template' in template_name:
            template = cv2.imread('templates/red_wolves_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(red_wolves_crop, mini_map_img, template)
        if 'red_chickens_template' in template_name:
            template = cv2.imread('templates/red_chickens_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(red_chickens_crop, mini_map_img, template)
        if 'red_red_buff_template' in template_name:
            template = cv2.imread('templates/red_red_buff_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(red_red_buff_crop , mini_map_img, template)
        if 'red_krugs_template' in template_name:
            template = cv2.imread('templates/red_krugs_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(red_krugs_crop, mini_map_img, template)

        if 'blue_toad_template' in template_name:
            template = cv2.imread('templates/blue_toad_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(blue_toad_crop, mini_map_img, template)
        if 'blue_blue_buff_template' in template_name:
            template = cv2.imread('templates/blue_blue_buff_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(blue_blue_buff_crop, mini_map_img, template)
        if 'blue_wolves_template' in template_name:
            template = cv2.imread('templates/blue_wolves_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(blue_wolves_crop, mini_map_img, template)
        if 'blue_chickens_template' in template_name:
            template = cv2.imread('templates/blue_chickens_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(blue_chickens_crop, mini_map_img, template)
        if 'blue_red_buff_crop' in template_name:
            template = cv2.imread('templates/blue_red_buff_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(blue_red_buff_crop, mini_map_img, template)
        if 'blue_krugs_template' in template_name:
            template = cv2.imread('templates/blue_krugs_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(blue_krugs_crop, mini_map_img, template)

        if 'top_side_scuttle_template' in template_name:
            template = cv2.imread('templates/top_side_scuttle_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(top_side_scuttle_crop, mini_map_img, template)
        if 'bot_side_scuttle_template' in template_name:
            template = cv2.imread('templates/bot_side_scuttle_template.png', cv2.IMREAD_COLOR)
            box = do_template_match(bot_side_scuttle_crop, mini_map_img, template)



if __name__ == '__main__':
    for image_path in sort_files_numerically('frames/'):
        camps_seen = camps_seen_for_frame('frames/' + image_path)
