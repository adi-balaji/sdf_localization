import open3d as o3d
import torch
import os
import pytorch_volumetric as pv
from pytorch_volumetric import sample_mesh_points
from pytorch_volumetric import voxel
from pytorch_volumetric import sdf
import numpy as np
import matplotlib.pyplot as plt
import time

OBJS_DIR = "/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/objs/"
PCD_DIR = "/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/pcd/"
RESOLUTION = 0.005
DEVICE = "cpu"
STEP_SIZE = 1e-4
STEP_SIZE_ROT = 1e-5

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="SDF Localization")

GT_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "drill.pcd"))
GT_pcd.paint_uniform_color([0.0, 0.0, 0.9])
updated_pcd = o3d.geometry.PointCloud() #intialize scene point cloud
vis.add_geometry(GT_pcd)
vis.add_geometry(updated_pcd)

obj = pv.MeshObjectFactory(os.path.join(OBJS_DIR, "drill.obj")) #create mesh object for PV
drill_sdf = pv.MeshSDF(obj) # compute SDF for the mesh object

drill_view_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "transformed_drill.pcd")) #
drill_np = np.asarray(drill_view_pcd.points)
drill_pcd_tensor = torch.tensor(drill_np, dtype=torch.float32) #convert scene point cloud to tensor

t = torch.zeros(3)
R = torch.eye(3)

# objective minimize function F = summation drill_sdf(T(p_i, R, T))^2
while True:
    # Query SDF for current transformed points
    t_pcd = drill_pcd_tensor @ R + t.T  # Apply the transformation
    sdf_vals_tr, sdf_grads_tr = drill_sdf(t_pcd)  # SDF values and gradients

    # Compute dF/dt
    dF_dt = 2 * (sdf_vals_tr[:, None] * sdf_grads_tr).sum(dim=0)

    # Compute dF/dR
    dF_dR = 2 * (torch.sum(sdf_vals_tr, dim=0)) * (sdf_grads_tr.T.mm(drill_pcd_tensor))

    # Update translation t
    t = t - STEP_SIZE * dF_dt

    dR = STEP_SIZE_ROT * dF_dR
    U, _, V = torch.svd(R + dR)
    R = U @ V.T


    updated_pcd.points = o3d.utility.Vector3dVector(t_pcd.numpy())
    vis.update_geometry(updated_pcd)
    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()
