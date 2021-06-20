import pickle
import json
import numpy as np
import math
import os
from pyquaternion import Quaternion
import time

from nuscenes.nuscenes import NuScenes


test_night_scene_tokens = ['e59a4d0cc6a84ed59f78fb21a45cdcb4',
                           '7209495d06f24712a063ac6c4a9b403b',
                           '3d776ea805f240bb925bd9b50b258416',
                           '48f81c548d0148fc8010a73d70b2ef9c',
                           '2ab683f384234dce89800049dec19a30',
                           '7edca4c44eac4f52a3105e1794e56b7e',
                           '81c939ce8c0d4cc7b159cb5ed4c4e712',
                           '24e6e64ecf794be4a51f7454c8b6d0b2',
                           '828ed34a5e0c456fbf0751cabbab3341',
                           'edfd6cfd1805477fbeadbd29f39ed599',
                           '7692a3e112b44b408d191e45954a813c',
                           '58d27a9f83294d99a4ff451dcad5f4d2',
                           'a1573aef0bf74324b373dd8a22b4dd68',
                           'ba06095d4e2e425b8e398668abc301d8',
                           '7c315a1db2ac49439d281605f3cca6be',
                           '732d7a84353f4ada803a9a115728496c',
                           '1630a1d9cf8a46b3843662a23126e3f6',
                           'f437809584344859882bdff7f8784c43']


def get_scene_lidar_token(nusc, scene_token, frame_skip=2):
    sensor = 'LIDAR_TOP'
    scene = nusc.get('scene', scene_token)
    first_sample = nusc.get('sample', scene['first_sample_token'])
    lidar = nusc.get('sample_data', first_sample['data'][sensor])

    lidar_token_list = [lidar['token']]
    counter = 1
    while lidar['next'] != '':
        lidar = nusc.get('sample_data', lidar['next'])
        counter += 1
        if counter % frame_skip == 0:
            lidar_token_list.append(lidar['token'])
    return lidar_token_list


def get_lidar_token_list(nusc, frame_skip):
    daytime_scene_list = []
    for scene in nusc.scene:
        if 'night' in scene['description'] \
                or 'Night' in scene['description'] \
                or scene['token'] in test_night_scene_tokens:
            continue
        else:
            daytime_scene_list.append(scene['token'])

    lidar_token_list = []
    for scene_token in daytime_scene_list:
        lidar_token_list += get_scene_lidar_token(nusc, scene_token, frame_skip=frame_skip)
    return lidar_token_list


def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P


def get_sample_data_ego_pose_P(nusc, sample_data):
    sample_data_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
    sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
    sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
    return sample_data_pose_P


def search_nearby_cameras(nusc,
                          init_camera,
                          max_translation,
                          direction,
                          lidar_P_inv,
                          nearby_camera_token_list):
    init_camera_direction_token = init_camera[direction]
    if init_camera_direction_token == '':
        return nearby_camera_token_list

    camera = nusc.get('sample_data', init_camera_direction_token)
    while True:
        camera_token = camera[direction]
        if camera_token == '':
            break
        camera = nusc.get('sample_data', camera_token)
        camera_P = get_sample_data_ego_pose_P(nusc, camera)
        P_lc = np.dot(lidar_P_inv, camera_P)
        t_lc = P_lc[0:3, 3]
        t_lc_norm = np.linalg.norm(t_lc)

        if t_lc_norm < max_translation:
            nearby_camera_token_list.append(camera_token)
        else:
            break
    return nearby_camera_token_list


def get_nearby_camera_token_list(nusc,
                                 lidar_token,
                                 max_translation,
                                 camera_name):
    lidar = nusc.get('sample_data', lidar_token)
    lidar_P = get_sample_data_ego_pose_P(nusc, lidar)
    lidar_P_inv = np.linalg.inv(lidar_P)

    lidar_sample_token = lidar['sample_token']
    lidar_sample = nusc.get('sample', lidar_sample_token)

    init_camera_token = lidar_sample['data'][camera_name]
    init_camera = nusc.get('sample_data', init_camera_token)
    nearby_camera_token_list = [init_camera_token]

    nearby_camera_token_list = search_nearby_cameras(
        nusc,
        init_camera,
        max_translation,
        'next',
        lidar_P_inv,
        nearby_camera_token_list)
    nearby_camera_token_list = search_nearby_cameras(
        nusc,
        init_camera,
        max_translation,
        'prev',
        lidar_P_inv,
        nearby_camera_token_list)

    return nearby_camera_token_list


def get_nearby_camera(nusc, lidar_token, max_translation):
    cam_list = ['CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT']
    nearby_cam_token_dict = {}
    for camera_name in cam_list:
        nearby_cam_token_dict[camera_name] \
            = get_nearby_camera_token_list(nusc,
                                           lidar_token,
                                           max_translation,
                                           camera_name)
    return nearby_cam_token_dict


def make_nuscenes_dataset(nusc, frame_skip, max_translation):
    dataset = []

    lidar_token_list = get_lidar_token_list(nusc,
                                            frame_skip)
    for i, lidar_token in enumerate(lidar_token_list):
        # begin_t = time.time()
        nearby_camera_token_dict = get_nearby_camera(nusc,
                                                     lidar_token,
                                                     max_translation)

        dataset.append((lidar_token, nearby_camera_token_dict))

        # print('lidar %s takes %f' % (lidar_token, time.time()-begin_t))
        if i % 100 == 0:
            print('%d done...' % i)

    return dataset


def load_dataset_info(filepath):
    with open(filepath, 'rb') as f:
        dataset_read = pickle.load(f)
    return dataset_read


def main():
    root_path = '/extssd/jiaxin/nuscenes/test'
    nusc = NuScenes(version='v1.0-test', dataroot=root_path, verbose=True)

    begin_t = time.time()
    dataset = make_nuscenes_dataset(nusc,
                                    frame_skip=2,
                                    max_translation=10)
    print('takes %f' % (time.time() - begin_t))
    print(len(dataset))

    output_file = os.path.join(root_path, 'dataset_info.list')
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
