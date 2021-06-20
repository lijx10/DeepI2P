import numpy as np
import os
import open3d
import cv2


def coordinate_cam_to_NWU(pc_np):
    assert pc_np.shape[0] == 3
    pc_nwu_np = np.copy(pc_np)
    pc_nwu_np[0, :] = pc_np[2, :]  # x <- z
    pc_nwu_np[1, :] = -pc_np[0, :]  # y <- -x
    pc_nwu_np[2, :] = -pc_np[1, :]  # z <- -y
    return pc_nwu_np


def coordinate_NWU_to_cam(pc_np):
    assert pc_np.shape[0] == 3
    pc_cam_np = np.copy(pc_np)
    pc_cam_np[0, :] = -pc_np[1, :]  # x <- -y
    pc_cam_np[1, :] = -pc_np[2, :]  # y <- -z
    pc_cam_np[2, :] = pc_np[0, :]  # z <- x
    return pc_cam_np


class KittiCalibHelper:
    def __init__(self, root_path):
        self.root_path = root_path
        self.calib_matrix_dict = self.read_calib_files()

    def read_calib_files(self):
        seq_folders = [name for name in os.listdir(os.path.join(self.root_path, 'calib'))]
        calib_matrix_dict = {}
        for seq in seq_folders:
            calib_file_path = os.path.join(self.root_path, 'calib', seq, 'calib.txt')
            with open(calib_file_path, 'r') as f:
                for line in f.readlines():
                    seq_int = int(seq)
                    if calib_matrix_dict.get(seq_int) is None:
                        calib_matrix_dict[seq_int] = {}

                    key = line[0:2]
                    mat = np.fromstring(line[4:], sep=' ').reshape((3, 4)).astype(np.float32)
                    if 'Tr' == key:
                        P = np.identity(4)
                        P[0:3, :] = mat
                        calib_matrix_dict[seq_int][key] = P
                    else:
                        K = mat[0:3, 0:3]
                        calib_matrix_dict[seq_int][key + '_K'] = K
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]
                        # mat[0, 3] = fx*tx + cx*tz
                        # mat[1, 3] = fy*ty + cy*tz
                        # mat[2, 3] = tz
                        tz = mat[2, 3]
                        tx = (mat[0, 3] - cx * tz) / fx
                        ty = (mat[1, 3] - cy * tz) / fy
                        P = np.identity(4)
                        P[0:3, 3] = np.asarray([tx, ty, tz])
                        calib_matrix_dict[seq_int][key] = P

        return calib_matrix_dict

    def get_matrix(self, seq: int, matrix_key: str):
        return self.calib_matrix_dict[seq][matrix_key]

    def transform_pc_vel_to_img(self,
                                pc: np.ndarray,
                                seq: int = 0,
                                img_key: str = 'P2',
                                Pi: np.ndarray=None,
                                Tr: np.ndarray=None):
        """

        :param pc: 3xN
        :param seq: int
        :param img_key: 'P0', 'P1', 'P2', 'P3'
        :return: 3xN
        """
        pc_homo = np.concatenate((pc, np.ones((1, pc.shape[1]))), axis=0)  # 3xN
        if Pi is None:
            Pi = self.get_matrix(seq, img_key)
        if Tr is None:
            Tr = self.get_matrix(seq, 'Tr')
        pc_img_homo = np.dot(np.dot(Pi, Tr), pc_homo)  # 4x4 * 4x4 * 4xN
        return pc_img_homo[0:3, :]

    def transform_pc_img_to_vel(self,
                                pc: np.ndarray,
                                seq: int = 0,
                                img_key: str = 'P2',
                                Pi: np.ndarray=None,
                                Tr: np.ndarray=None):
        """

        :param pc: 3xN
        :param seq: int
        :param img_key: 'P0', 'P1', 'P2', 'P3'
        :return: 3xN
        """
        pc_homo = np.concatenate((pc, np.ones((1, pc.shape[1]))), axis=0)  # 3xN
        if Pi is None:
            Pi_inv = np.linalg.inv(self.get_matrix(seq, img_key))
        else:
            Pi_inv = np.linalg.inv(Pi)
        if Tr is None:
            Tr_inv = np.linalg.inv(self.get_matrix(seq, 'Tr'))
        else:
            Tr_inv = np.linalg.inv(Tr)
        pc_vel_homo = np.dot(np.dot(Tr_inv, Pi_inv), pc_homo)  # 4x4 * 4x4 * 4xN
        return pc_vel_homo[0:3, :]


def draw_points_on_img(pc_np, img):
    """

    :param pc_np:
    :param img:
    :return:
    """
    img_vis = np.copy(img)
    H, W = img.shape[0], img.shape[1]

    if pc_np.shape[0] == 3:
        pc_pixels = pc_np[0:2, :] / pc_np[2:3, :]
    else:
        pc_pixels = pc_np
    for i in range(pc_pixels.shape[1]):
        px = int(pc_pixels[0, i])
        py = int(pc_pixels[1, i])
        # determine a point on image plane
        if px>=0 and px<=W-1 and py>=0 and py<=H-1:
            cv2.circle(img_vis, (px, py), 1, (255, 0, 0), -1)
        elif px<0 or px>W or py<0 or py>H:
            print('(px, py): (%d, %d)' % (px, py))
            assert False
    return img_vis


def projection_pc_img(pc_np, img, K, size=2):
    """

    :param pc_np: points in camera coordinate
    :param img: image of the same frame
    :param K: Intrinsic matrix
    :return:
    """
    img_vis = np.copy(img)
    H, W = img.shape[0], img.shape[1]

    pc_np_front = pc_np[:, pc_np[2, :]>1.0]  # 3xN

    pc_pixels = np.dot(K, pc_np_front)  # 3xN
    pc_pixels = pc_pixels / pc_pixels[2:, :]  # 3xN
    for i in range(pc_pixels.shape[1]):
        px = int(pc_pixels[0, i])
        py = int(pc_pixels[1, i])
        # determine a point on image plane
        if px>=0 and px<=W-1 and py>=0 and py<=H-1:
            cv2.circle(img_vis, (px, py), size, (255, 0, 0), -1)
    return img_vis


def crop_pc_with_img(pc_np, intensity_np, sn_np, img, K):
    """

    :param pc_np:
    :param intensity_np:
    :param sn_np:
    :param img:
    :param K:
    :return:
    """
    H, W = img.shape[0], img.shape[1]

    pc_pixels = np.dot(K, pc_np)  # 3xN
    pc_pixels = pc_pixels / pc_pixels[2:, :]  # 3xN

    pc_pixels = np.round(pc_pixels)
    pc_mask_x = np.logical_and(pc_pixels[0, :] >= 0, pc_pixels[0, :] <= W - 1)
    pc_mask_y = np.logical_and(pc_pixels[1, :] >= 0, pc_pixels[1, :] <= H - 1)
    pc_mask = np.logical_and(pc_mask_x, pc_mask_y)

    pc_np_img = pc_np[:, pc_mask]
    intensity_np_img = intensity_np[:, pc_mask]
    sn_np_img = sn_np[:, pc_mask]

    return pc_np_img, intensity_np_img, sn_np_img


def camera_matrix_cropping(K: np.ndarray, dx: float, dy: float):
    K_crop = np.copy(K)
    K_crop[0, 2] -= dx
    K_crop[1, 2] -= dy
    return K_crop


def camera_matrix_scaling(K: np.ndarray, s: float):
    K_scale = s * K
    K_scale[2, 2] = 1
    return K_scale


class ProjectiveFarthestSampler:
    def __init__(self):
        self.fps_2d = FarthestSampler(dim=2)

    def sample(self, pts, k, projection_K):
        # 1. project the points onto image with projection K
        pts_2d = np.dot(projection_K, pts)  # 3x3 * 3xN -> 3xN
        pts_2d = pts_2d[0:2, :] / pts_2d[2:, :]  # 2xN

        # 2. FPS on 2d
        nodes_2d, nodes_idx = self.fps_2d.sample(pts_2d, k)

        # 3. get the corresponding 3d points
        nodes_3d = pts[:, nodes_idx]

        return nodes_3d, nodes_idx


class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx


def voxel_downsample(pc_np, voxel_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pc_np))
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.transpose(np.asarray(downpcd.points, dtype=np.float32))


def fps_approximate(pc_np, voxel_size, node_num):
    pc_down_np = voxel_downsample(pc_np, voxel_size)
    while pc_down_np.shape[1] < node_num:
        voxel_size *= 0.75
        pc_down_np = voxel_downsample(pc_np, voxel_size)
    return pc_down_np[:, np.random.choice(pc_down_np.shape[1], int(node_num), replace=False)]
