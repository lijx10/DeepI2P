import os
import shutil
from distutils.dir_util import copy_tree


if __name__ == '__main__':
    src_folder = '/ssd/jiaxin/datasets/kitti/data_odometry_velodyne_NWU/sequences'
    sensor = 'stride4-acc50-voxel0.4'
    dst_folder = '/home/jiaxin/remote/datasets/kitti/data_odometry_velodyne_NWU/sequences'

    for seq in range(11):
        src_sensor_folder = os.path.join(src_folder, '%02d' % seq, sensor)
        dst_sensor_folder = os.path.join(dst_folder, '%02d' % seq, sensor)
        if not os.path.exists(dst_sensor_folder):
            os.mkdir(dst_sensor_folder)
        copy_tree(src_sensor_folder, dst_sensor_folder)
        print(dst_sensor_folder)
