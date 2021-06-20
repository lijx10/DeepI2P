import math
import os
import cv2
import time


def wrap_in_pi(x):
    x = math.fmod(x+math.pi, math.pi*2)
    if x<0:
        x += math.pi*2
    return x - math.pi


def load_images(folder_path):
    img_list = []

    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    counter = 0
    begin_t = time.time()
    for file in file_list:
        img = cv2.imread(file)
        img_list.append(img)
        counter += 1
    print("average load time: %f" % ((time.time() - begin_t) / counter))
    return img_list


if __name__ == '__main__':
    img_list = load_images('/extssd/jiaxin/oxford/2014-12-16-09-14-09/stereo/centre')
    print(len(img_list))