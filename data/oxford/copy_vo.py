import os
import shutil
from distutils.dir_util import copy_tree


if __name__ == '__main__':
    oxford_folder = '/data/datasets/oxford'
    sensor = 'vo'
    src_folder = '/data/datasets/oxford-%s/%s' % (sensor, sensor)

    filename_list = [f for f in os.listdir(oxford_folder) if os.path.isdir(os.path.join(oxford_folder, f))]
    filename_list = list(set(filename_list))
    filename_list.sort()

    for traversal in filename_list:
        if not os.path.exists(os.path.join(src_folder, traversal, sensor)):
            print("not exist:")
            print(os.path.join(src_folder, traversal, sensor))
        else:
            copy_tree(os.path.join(src_folder, traversal), os.path.join(oxford_folder, traversal))