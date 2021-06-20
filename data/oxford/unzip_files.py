import os
import tarfile


if __name__ == '__main__':
    # tar_folder = '/media/jiaxin/6D7ED1DF490F0D2B/zijian'
    tar_folder = '/sunlissd/jiaxin/oxford_tar'
    # output_folder = '/data/datasets/oxford'
    output_folder = '/extssd/jiaxin/oxford-lms-raw'

    filename_list = [f[0:19] for f in os.listdir(tar_folder) if os.path.isfile(os.path.join(tar_folder, f))]
    filename_list = list(set(filename_list))
    filename_list.sort()

    # sensor_list = ['gps', 'lms_front', 'stereo_centre', 'vo', 'tags']
    sensor_list = ['lms_front']
    for traversal in filename_list:
        # if exist, skip
        if os.path.exists(os.path.join(output_folder, traversal)):
            continue

        for sensor in sensor_list:
            # single file
            tar_path = os.path.join(tar_folder, traversal+'_'+sensor+'.tar')
            if os.path.exists(tar_path):
                print(tar_path)
                tar_file = tarfile.open(tar_path, 'r')
                tar_file.extractall(path=output_folder)
                tar_file.close()

            # multiple files
            for i in range(1, 99):
                tar_path = os.path.join(tar_folder, traversal + '_' + sensor + '_%02d.tar' % i)
                if os.path.exists(tar_path):
                    print(tar_path)
                    tar_file = tarfile.open(tar_path, 'r')
                    tar_file.extractall(path=output_folder)
                    tar_file.close()
                else:
                    break