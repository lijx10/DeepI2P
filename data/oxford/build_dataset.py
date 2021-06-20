import numpy as np
import numpy.matlib as ml
import os
import math
import time
import cv2
import open3d
import shutil
import sys
import re
import csv
import bisect
import multiprocessing

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

project_absolute_path = '/ssd/jiaxin/point-img-feature'
sys.path.append(os.path.join(project_absolute_path, "data/oxford/robotcar_dataset_sdk/python"))
from data.oxford.robotcar_dataset_sdk.python.build_pointcloud import build_pointcloud
from data.oxford.robotcar_dataset_sdk.python.transform import build_se3_transform
from data.oxford.robotcar_dataset_sdk.python.image import load_image
from data.oxford.robotcar_dataset_sdk.python.camera_model import CameraModel
from data.oxford.robotcar_dataset_sdk.python.interpolate_poses import interpolate_poses, interpolate_vo_poses




from util import vis_tools


class VOManager:
    def __init__(self, vo_path: str):
        """
        :param vo_path (str): path to file containing relative poses from visual odometry.
        """
        with open(vo_path) as vo_file:
            vo_reader = csv.reader(vo_file)
            headers = next(vo_file)

            self.counter = 0
            self.timestamp_list = []
            self.xyzrpy_list = []
            for row in vo_reader:
                timestamp = int(row[0])
                xyzrpy = [float(v) for v in row[2:8]]
                self.timestamp_list.append(timestamp)
                self.xyzrpy_list.append(xyzrpy)
                self.counter += 1

    def interpolate_vo_poses(self, pose_timestamps, origin_timestamp):
        pose_timestamps = pose_timestamps.copy()
        lower_timestamp = min(min(pose_timestamps), origin_timestamp)
        upper_timestamp = max(max(pose_timestamps), origin_timestamp)

        lower_timestamp_idx = bisect.bisect_left(self.timestamp_list, lower_timestamp)
        lower_timestamp_idx -= 1
        lower_timestamp_idx = max(0, lower_timestamp_idx)

        upper_timestamp_idx = bisect.bisect_left(self.timestamp_list, upper_timestamp)
        upper_timestamp_idx = min(upper_timestamp_idx, self.counter-1)

        vo_timestamps = [self.timestamp_list[lower_timestamp_idx]]
        abs_poses = [ml.identity(4)]
        for idx in range(lower_timestamp_idx+1, upper_timestamp_idx+1):
            vo_timestamps.append(self.timestamp_list[idx])

            xyzrpy = self.xyzrpy_list[idx]
            rel_pose = build_se3_transform(xyzrpy)
            abs_pose = abs_poses[-1] * rel_pose
            abs_poses.append(abs_pose)

        return interpolate_poses(vo_timestamps, abs_poses, pose_timestamps, origin_timestamp)


def my_build_pointcloud(G_posesource_laser, lidar_dir, vo_manager,
                        timestamps, origin_time, skip_threshold=None, remove_ground_threshold=None):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        lidar_dir (str): Directory containing LIDAR scans.
        poses_file (str): Path to a file containing pose information. Can be VO or INS data.
        extrinsics_dir (str): Directory containing extrinsic calibrations.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        numpy.ndarray: 3xn array of (x, y, z) coordinates of pointcloud
        numpy.array: array of n reflectance values or None if no reflectance values are recorded (LDMRS)

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")

    # sensor is VO, which is located at the main vehicle frame
    poses = vo_manager.interpolate_vo_poses(timestamps, origin_time)

    pointcloud = np.array([[0], [0], [0], [0]])
    reflectance = np.empty((0))

    previous_pose = None
    skip_counter = 0
    for i in range(0, len(poses)):
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.bin')

        if not os.path.isfile(scan_path):
            continue

        # if the car doesn't move much, skip
        if (previous_pose is not None) and (skip_threshold is not None):
            delta_pose = np.dot(np.linalg.inv(previous_pose), poses[i])
            delta_translation_norm = np.linalg.norm(delta_pose[0:3, 3])
            if delta_translation_norm < skip_threshold:
                skip_counter += 1
                continue

        # scan_file = open(scan_path)
        scan = np.fromfile(scan_path, np.double)
        # scan_file.close()

        scan = scan.reshape((len(scan) // 3, 3)).transpose()  # 3xN
        # x is pointing to the ground
        if remove_ground_threshold is not None and remove_ground_threshold > -1:
            foreground_mask = scan[0, :] < remove_ground_threshold
            scan = scan[:, foreground_mask]

        # LMS scans are tuples of (x, y, reflectance)
        reflectance = np.concatenate((reflectance, np.ravel(scan[2, :])))
        scan[2, :] = np.zeros((1, scan.shape[1]))

        scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))
        pointcloud = np.hstack([pointcloud, scan])
        previous_pose = poses[i]

    pointcloud = pointcloud[:, 1:]
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return pointcloud, reflectance, skip_counter


def downsample(pointcloud, reflectance, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    reflectance_max = np.max(reflectance)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0] = reflectance / reflectance_max
    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = np.concatenate((down_pcd_points,
                                 np.ones((1, down_pcd_points.shape[1]))),
                                axis=0)  # 4xN
    reflectance = np.asarray(down_pcd.colors)[:, 0] * reflectance_max

    return pointcloud, reflectance


def save_pc_img_for_traversal(traversal,
                              oxford_raw_lms_front_root,
                              oxford_raw_root,
                              oxford_output_root,
                              models_dir,
                              extrinsics_dir,
                              is_build_pc,
                              remove_ground_threshold,
                              is_build_img,
                              is_plot,
                              is_debug,
                              pc_sample_distance,
                              min_vehicle_velocity,
                              accumulation_distance,
                              ignore_first_n_second,
                              voxel_grid_downsample_size):
    print("--- %s ---" % traversal)
    # debug
    # if traversal != '2015-02-13-09-16-26':
    #     continue

    # camera configuration -------------------------------
    image_dir = os.path.join(oxford_raw_root, traversal, 'stereo', 'centre')
    model = CameraModel(models_dir, image_dir)
    G_camera_image_inv = np.linalg.inv(model.G_camera_image)
    print('%s camera model: ' % traversal, model.camera)
    with open(os.path.join(extrinsics_dir, model.camera + '.txt')) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
    G_camera_vehicle = build_se3_transform(extrinsics)
    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle

    # load the camera timestamps
    camera_timestamps_path = os.path.join(image_dir, os.pardir, model.camera + '.timestamps')
    if not os.path.isfile(camera_timestamps_path):
        camera_timestamps_path = os.path.join(image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

    camera_timestamp_list = []
    with open(camera_timestamps_path) as camera_timestamps_file:
        for i, line in enumerate(camera_timestamps_file):
            camera_timestamp_list.append(int(line.split(' ')[0]))

    # lidar configuration ----------------------------------
    lidar_dir = os.path.join(oxford_raw_lms_front_root, traversal, 'lms_front')
    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', lidar_dir).group(0)
    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    # load lidar timestamps
    lidar_timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')
    lidar_timestamps_list = []
    with open(lidar_timestamps_path) as lidar_timestamps_file:
        for line in lidar_timestamps_file:
            lidar_timestamp = int(line.split(' ')[0])
            lidar_timestamps_list.append(lidar_timestamp)

    # VO pose -----------------------------------------------
    poses_file = os.path.join(oxford_raw_root, traversal, 'vo', 'vo.csv')
    vo_manager = VOManager(poses_file)

    # build a point cloud every 1 meter
    # one oxford route is 10km, so there will be 10k point clouds in a route
    # 1. identity the timestamp after ignore_first_n_second
    init_idx = 0
    for idx, camera_timestamp in enumerate(camera_timestamp_list):
        if camera_timestamp - camera_timestamp_list[0] > ignore_first_n_second * 1e6:
            init_idx = idx
            break
    print('%s len(camera_timestamp_list) = %d, init_idx = %d' % (traversal, len(camera_timestamp_list), init_idx))
    camera_timestamp_list = camera_timestamp_list[init_idx:]

    if is_build_pc:
        if remove_ground_threshold is not None:
            output_pc_folder = os.path.join(oxford_output_root, traversal, 'lms_front_foreground')
        else:
            output_pc_folder = os.path.join(oxford_output_root, traversal, 'lms_front')
        if not os.path.exists(output_pc_folder):
            os.makedirs(output_pc_folder)

        # 2. for loop to find the all timestamps of every "pc_sample_distance" meter location
        camera_per_meter_idx_list = [0]
        for idx in range(1, len(camera_timestamp_list), 1):
            relative_pose = vo_manager.interpolate_vo_poses([camera_timestamp_list[idx]],
                                                            camera_timestamp_list[camera_per_meter_idx_list[-1]])
            relative_translation = np.linalg.norm(relative_pose[0][0:3, 3])

            # debug
            # print('idx %d - relative_translation %f' % (idx, relative_translation))
            if relative_translation >= pc_sample_distance:
                camera_per_meter_idx_list.append(idx)

            # debug
            if is_debug and idx > 3000:
                break

        # 3. build point cloud for nodes in camera_per_meter_idx_list,
        # if this node have enough back & forward travel distance
        print("%s len(camera_per_meter_idx_list) = %d" % (traversal, len(camera_per_meter_idx_list)))
        idx_margin = math.ceil(0.5 * accumulation_distance / pc_sample_distance)
        pc_center_timestamp_list = []
        for i in range(idx_margin, len(camera_per_meter_idx_list) - idx_margin, 1):
            back_idx = camera_per_meter_idx_list[i - idx_margin]
            back_timestamp = camera_timestamp_list[back_idx]

            center_idx = camera_per_meter_idx_list[i]
            center_timestamp = camera_timestamp_list[center_idx]

            front_idx = camera_per_meter_idx_list[i + idx_margin]
            front_timestamp = camera_timestamp_list[front_idx]

            # search lidar_timestamps_all, to find timestamps that is [start_time, end_time]
            left_idx = bisect.bisect_left(lidar_timestamps_list, back_timestamp)
            right_idx = bisect.bisect_right(lidar_timestamps_list, front_timestamp)
            accumulate_timestamps = lidar_timestamps_list[left_idx:right_idx]

            # lidar is 50Hz, so we can calculate the vehicle velocity during this accumulation
            # if velocity > min_vehicle_velocity, skip
            vehicle_velocity = accumulation_distance / (len(accumulate_timestamps) / 50)
            if vehicle_velocity < min_vehicle_velocity:
                print('[PC] %s Velocity %.2f, accumulated frame number %d, skip.' % (traversal,
                                                                                     vehicle_velocity,
                                                                                     len(accumulate_timestamps)))
                continue

            if os.path.isfile(os.path.join(output_pc_folder, str(center_timestamp) + '.npy')):
                continue

            begin_t = time.time()
            # 3xN, N
            pointcloud, reflectance, skip_counter = my_build_pointcloud(G_posesource_laser,
                                                                        lidar_dir,
                                                                        vo_manager,
                                                                        accumulate_timestamps,
                                                                        center_timestamp,
                                                                        skip_threshold=voxel_grid_downsample_size/16.0,
                                                                        remove_ground_threshold=remove_ground_threshold)
            # downsample point cloud by voxel grid
            pointcloud, reflectance = downsample(pointcloud, reflectance, voxel_grid_downsample_size)
            end_t = time.time()

            pointcloud = np.dot(np.dot(G_camera_image_inv, G_camera_posesource), pointcloud)
            print("[PC] %s Velocity %.2f, accumulated %d/%d frames, "
                  "pc has %d downsampled points, time consumed %.2f" % (traversal,
                                                                        vehicle_velocity,
                                                                        len(accumulate_timestamps) - skip_counter,
                                                                        len(accumulate_timestamps),
                                                                        pointcloud.shape[1],
                                                                        end_t - begin_t))

            output_pc = np.concatenate((pointcloud[0:3, :],
                                        np.expand_dims(reflectance, axis=0)),
                                       axis=0).astype(np.float32)
            np.save(os.path.join(output_pc_folder, str(center_timestamp) + '.npy'), output_pc)
            pc_center_timestamp_list.append(center_timestamp)

            if is_plot:
                pointcloud = output_pc
                pointcloud[3, :] = 1

                image_path = os.path.join(image_dir, str(center_timestamp) + '.png')
                image = load_image(image_path, model)

                uv, depth = model.project(pointcloud, image.shape, is_need_transform=False)

                plt.imshow(image)
                plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
                plt.xlim(0, image.shape[1])
                plt.ylim(image.shape[0], 0)
                plt.xticks([])
                plt.yticks([])

                vis_tools.plot_pc(output_pc[0:3, :])

                plt.show()

        # build point cloud pose
        begin_t = time.time()
        pc_poses_init_i = vo_manager.interpolate_vo_poses(pc_center_timestamp_list,
                                                          camera_timestamp_list[0])
        print('[PC] build poses takes %.2f' % (time.time() - begin_t))
        np.save(os.path.join(oxford_output_root, traversal, 'pc_timestamps.npy'),
                np.asarray(pc_center_timestamp_list, dtype=np.int))
        np.save(os.path.join(oxford_output_root, traversal, 'pc_poses.npy'),
                np.asarray(pc_poses_init_i, dtype=np.float))
        print("[PC] save %d timestamps, %d poses." % (len(pc_center_timestamp_list), len(pc_poses_init_i)))

    if is_build_img:
        # build image
        output_image_folder = os.path.join(oxford_output_root, traversal, 'stereo', 'centre')
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)

        valid_camera_timestamp_list = camera_timestamp_list.copy()
        for idx in range(0, len(camera_timestamp_list), 1):
            output_img_path = os.path.join(output_image_folder, str(camera_timestamp_list[idx]) + '.jpg')
            # skip images that are already decoded
            if os.path.isfile(output_img_path):
                continue

            input_img_path = os.path.join(image_dir, str(camera_timestamp_list[idx]) + '.png')
            # is image doesn't exist, skip
            if not os.path.isfile(input_img_path):
                valid_camera_timestamp_list.remove(camera_timestamp_list[idx])
                continue

            image = load_image(input_img_path, model)
            cv2.imwrite(output_img_path,
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        camera_timestamp_list = valid_camera_timestamp_list

        # build image pose for each timestamp
        print("len(camera_timestamp_list): ", len(camera_timestamp_list))
        begin_t = time.time()
        camera_poses_init_i = vo_manager.interpolate_vo_poses(camera_timestamp_list,
                                                              camera_timestamp_list[0])
        print('[Image] build poses takes %.2f' % (time.time() - begin_t))
        np.save(os.path.join(oxford_output_root, traversal, 'camera_timestamps.npy'),
                np.asarray(camera_timestamp_list, dtype=np.int))
        np.save(os.path.join(oxford_output_root, traversal, 'camera_poses.npy'),
                np.asarray(camera_poses_init_i, dtype=np.float))
        print("[Image] save %d timestamps, %d poses." % (len(camera_timestamp_list), len(camera_poses_init_i)))


def read_tags_csv(csv_path):
    with open(csv_path) as csvfile:
        tags_reader = csv.reader(csvfile, delimiter=',')
        tags = []
        for row in tags_reader:
            tags += row
    return tags


def main():
    # ===============================================
    oxford_raw_lms_front_root = '/extssd/jiaxin/oxford-lms-raw'
    oxford_raw_root = '/data/datasets/oxford/oxford-extracted'
    oxford_output_root = '/extssd/jiaxin/oxford'
    models_dir = os.path.join(project_absolute_path, 'data/oxford/robotcar_dataset_sdk/models')
    extrinsics_dir = os.path.join(project_absolute_path, 'data/oxford/robotcar_dataset_sdk/extrinsics')

    is_build_pc = True
    remove_ground_threshold = 0.1
    is_build_img = False
    is_plot = False
    is_debug = False

    pc_sample_distance = 2
    min_vehicle_velocity = 0.2
    accumulation_distance = 100
    ignore_first_n_second = 20  # for VO and INS to initialize
    voxel_grid_downsample_size = 0.1

    thread_num = 42
    # ===============================================
    if is_debug:
        thread_num = 1

    raw_traversal_list = [f for f in os.listdir(oxford_raw_root) if os.path.isdir(os.path.join(oxford_raw_root, f))]
    raw_traversal_list.sort()

    # read csv files and delete night traversals
    night_traversal_list = []
    for traversal in raw_traversal_list:
        tags = read_tags_csv(os.path.join(oxford_raw_root, traversal, 'tags.csv'))
        if 'night' in tags:
            print(traversal + ' is night driving, skip.')
            night_traversal_list.append(traversal)
    if len(night_traversal_list) >= 1:
        for night_traversal in night_traversal_list:
            raw_traversal_list.remove(night_traversal)

    # skip traversals that are already done, "Done" is detected by the existence of tags.csv
    done_traversal_list = []
    for traversal in raw_traversal_list:
        if os.path.exists(os.path.join(oxford_output_root, traversal, 'tags.csv')):
            print(traversal + ' is already done, skip.')
            done_traversal_list.append(traversal)
    if len(done_traversal_list) >= 1:
        for done_traversal in done_traversal_list:
            raw_traversal_list.remove(done_traversal)

    batch_num = math.ceil(len(raw_traversal_list) / thread_num)
    for batch in range(batch_num):
        if batch == batch_num-1:
            batch_traversal_list = raw_traversal_list[batch*thread_num:]
        else:
            batch_traversal_list = raw_traversal_list[batch*thread_num : (batch+1)*thread_num]

        print("=== processing the following traversals in parallel")
        print(batch_traversal_list)

        threads = []
        for traversal in batch_traversal_list:
            threads.append(multiprocessing.Process(target=save_pc_img_for_traversal,
                                                   args=(traversal,
                                                         oxford_raw_lms_front_root,
                                                         oxford_raw_root,
                                                         oxford_output_root,
                                                         models_dir,
                                                         extrinsics_dir,
                                                         is_build_pc,
                                                         remove_ground_threshold,
                                                         is_build_img,
                                                         is_plot,
                                                         is_debug,
                                                         pc_sample_distance,
                                                         min_vehicle_velocity,
                                                         accumulation_distance,
                                                         ignore_first_n_second,
                                                         voxel_grid_downsample_size
                                                         )))
            # threads.append(multiprocessing.Process(target=None))

        for thread in threads:
            thread.start()

        for i, thread in enumerate(threads):
            thread.join()

            # copy tags as the final step
            traversal = batch_traversal_list[i]
            shutil.copy(os.path.join(oxford_raw_root, traversal, 'tags.csv'),
                        os.path.join(oxford_output_root, traversal, 'tags.csv'))



if __name__ == "__main__":
    main()