import os
import shutil
from distutils.dir_util import copy_tree


if __name__ == '__main__':
    oxford_folder = '/data/datasets/oxford'
    sensor = 'tags'
    src_folder = '/data/datasets/oxford-%s' % (sensor)

    filename_list = [f for f in os.listdir(oxford_folder) if os.path.isdir(os.path.join(oxford_folder, f))]
    filename_list = list(set(filename_list))
    filename_list.sort()

    for traversal in filename_list:
        tags_csv = os.path.join(src_folder, traversal, sensor+'.csv')
        dst_folder = os.path.join(oxford_folder, traversal)
        if not os.path.exists(tags_csv):
            print("not exist:")
            print(tags_csv)
        else:
            shutil.copyfile(tags_csv, os.path.join(dst_folder, sensor+'.csv'))
