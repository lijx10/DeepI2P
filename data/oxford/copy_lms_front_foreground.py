import os
import shutil
from distutils.dir_util import copy_tree


if __name__ == '__main__':
    src_folder = '/extssd/jiaxin/oxford'
    sensor = 'lms_front_foreground'
    dst_folder = '/home/jiaxin/nus/data/jiaxin/oxford'

    filename_list = [f for f in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, f))]
    filename_list = list(set(filename_list))
    filename_list.sort()

    for traversal in filename_list:
        if not os.path.exists(os.path.join(dst_folder, traversal, sensor)):
            dst_sensor_folder = os.path.join(dst_folder, traversal, sensor)
            print(dst_sensor_folder)
            os.mkdir(dst_sensor_folder)
            copy_tree(os.path.join(src_folder, traversal, sensor),
                      dst_sensor_folder)
