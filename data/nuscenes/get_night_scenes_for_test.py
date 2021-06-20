import pickle
import json
import numpy as np
import math
import os
from pyquaternion import Quaternion
import time
from PIL import Image

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes


def main():
    root_path = '/extssd/jiaxin/nuscenes/test'
    nusc = NuScenes(version='v1.0-test', dataroot=root_path, verbose=True)

    sensor = 'CAM_FRONT'

    counter = 0
    for i, scene in enumerate(nusc.scene):
        scene_token = scene['token']
        scene = nusc.get('scene', scene_token)
        first_sample = nusc.get('sample', scene['first_sample_token'])
        camera = nusc.get('sample_data', first_sample['data'][sensor])

        img = np.array(Image.open(os.path.join(nusc.dataroot, camera['filename'])).convert('L'))
        H, W = img.shape[0], img.shape[1]

        img_mean = np.mean(img.astype(np.float32))

        white_mask = img > 150
        white_area = np.sum(white_mask.astype(np.float32))

        if img_mean < 110 and white_area < (H*W)*0.1:
            print('\'%s\',' % (scene_token))
            counter += 1
            plt.figure()
            plt.gray()
            plt.imshow(img)
            plt.show()

    print('%d night scenes' % counter)

if __name__ == '__main__':
    main()