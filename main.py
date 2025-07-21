from ukf import UKFRegistration
import utils

import numpy as np
import open3d as o3d

# Load point cloud
pcd_target = o3d.io.read_point_cloud("data/bun_zipper.ply")
target = 1000*np.asarray(pcd_target.points).T   # in mm

# Independently downsample 
source = utils.downsample(target, sample_ratio=0.25)
target = utils.downsample(target, sample_ratio=0.25)

# Randomly transform source
max_t, max_theta = 5, 10
source, T_ref = utils.randomly_transformed(source, max_t=max_t, max_theta=max_theta)

# Run UKF registration
ukf_reg = UKFRegistration(alpha=0.25, 
                          beta=3, 
                          kappa=500, 
                          var_trans=max_t, 
                          var_theta=max_theta, 
                          isotropic=False, 
                          max_iter=100, 
                          threshold=0.3)
T_est, rmse = ukf_reg.register(source, target, add_noise=False)

# Transform source
aligned = utils.apply_transform(source, T_est)

# Compute 6 DOF error
e_theta, e_t = utils.evaluate(T_ref, T_est)

print("Estimated transformation:\n", T_est)
print("Final RMSE:", rmse)
print("Error (Rx, Ry, Rz): ", e_theta)
print("Error (tx, ty, tz): ", e_t)

# Visualize result
utils.visualize(target, source, aligned)
