import os
import shutil
from distutils.dir_util import copy_tree


if __name__ == '__main__':
    src_folder = '/data/datasets/oxford/oxford-extracted'
    sensor = 'lms_front'
    dst_folder = '/extssd/jiaxin/oxford-lms-raw'

    raw_traversal_list = [f for f in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, f))]
    raw_traversal_list.sort()

    for traversal in raw_traversal_list:
        print("--- %s ---" % traversal)
        target_traversal_folder = os.path.join(dst_folder, traversal)

        if os.path.exists(target_traversal_folder):
            print('%s already exist, skip.' % target_traversal_folder)
            continue

        os.mkdir(target_traversal_folder)
        copy_tree(os.path.join(src_folder, traversal, sensor), target_traversal_folder)
