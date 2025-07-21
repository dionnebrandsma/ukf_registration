import numpy as np
import open3d as o3d

from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

def find_correspondences(source: np.ndarray, target: np.ndarray):
    tree = KDTree(target.T)
    dist, idx = tree.query(source.T, k=1)
    return target[:, idx], dist

def compute_rmse(source, target):
    _, dist = find_correspondences(source, target)
    return np.sqrt(np.mean(np.square(dist)))

def apply_transform(points, T):
    ones = np.ones((1, points.shape[1]))
    homog = np.vstack((points, ones))
    transformed = T @ homog
    return transformed[:3]

def homogeneous_transformation(rot, trans):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T

def randomly_transformed(points, max_t, max_theta):
    random_trans = np.random.uniform(-max_t, max_t, 3)
    random_theta = np.random.uniform(-max_theta, max_theta, 3)
    
    random_R = R.from_euler('ZYX', np.deg2rad(random_theta)).as_matrix()
    random_T = homogeneous_transformation(random_R, random_trans)
    
    points = apply_transform(points.copy(), random_T)
    return points, random_T

def downsample(points, sample_ratio):
    pts = points.copy()

    n = int(np.size(pts, axis=1)*sample_ratio)
    idx = np.random.choice(np.size(pts, axis=1), n, replace=False)

    pts = pts[:, idx]
    return pts

def evaluate(T_ref, T_est):
    R_ref = T_ref[0:3,0:3]
    t_ref = T_ref[0:3,3]

    R_est = T_est[0:3,0:3]
    t_est = T_est[0:3,3]
    
    e_theta = R.from_matrix(R_est @ R_ref).as_euler('xyz', degrees=True)
    e_t = R_est @ t_ref + t_est
    return e_theta, e_t

def visualize(pts1, pts2, pts3=None):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1.T)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts2.T)

    pcd1.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])

    if pts3 is not None:
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(pts3.T)
        pcd3.paint_uniform_color([0, 0, 1])
        
        o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
    else:
        o3d.visualization.draw_geometries([pcd1, pcd2])