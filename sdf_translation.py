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
STEP_SIZE = 0.01

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

for i in range(200):

    sdf_vals, sdf_grads = drill_sdf(drill_pcd_tensor) #query sdf for sdf values and gradients
    t_new = STEP_SIZE * torch.mean(sdf_grads, dim=0) #update translation according to gradients
    t = t + t_new
    drill_pcd_tensor = drill_pcd_tensor - t_new #translate the scene point cloud

    #visualize
    updated_pcd.points = o3d.utility.Vector3dVector(drill_pcd_tensor.numpy())
    vis.update_geometry(updated_pcd)

    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()

print(t)

