import open3d as o3d
import torch
import os
import pytorch_volumetric as pv

import numpy as np
import matplotlib.pyplot as plt


OBJS_DIR = "/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/objs/"
PCD_DIR = "/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/pcd/"
RESOLUTION = 0.005
DEVICE = "cpu"
STEP_SIZE = 1e-3 # 1e-3
STEP_SIZE_ROT = 1e-4    # 1e-3
plot_loss = False

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="SDF Localization")

GT_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "drill.pcd"))
GT_pcd.paint_uniform_color([0.0, 0.0, 0.7])
updated_pcd = o3d.geometry.PointCloud()  # Initialize scene point cloud
vis.add_geometry(GT_pcd)
vis.add_geometry(updated_pcd)

obj = pv.MeshObjectFactory(os.path.join(OBJS_DIR, "drill.obj"))  # Create mesh object for PV
drill_sdf = pv.MeshSDF(obj)  # Compute SDF for the mesh object

drill_view_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "transformed_drill.pcd"))
R_test = drill_view_pcd.get_rotation_matrix_from_xyz((0., -0.0, 0.0))
drill_view_pcd.rotate(R_test, center=drill_view_pcd.get_center())

drill_np = np.asarray(drill_view_pcd.points)
drill_pcd_tensor = torch.tensor(drill_np, dtype=torch.float32)  # Convert scene point cloud to tensor

t = torch.zeros(3)
R = torch.eye(3)
losses = []

# Loss plotting setup
if plot_loss:
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot(losses, label="Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Real-Time Loss Plot")
    ax.legend()

# Objective function
while True:
    
    t_pcd = drill_pcd_tensor @ R + t.T  # transformation
    sdf_vals_tr, sdf_grads_tr = drill_sdf(t_pcd)  # SDF values and gradients

    # compute dF/dt
    dF_dt = 2 * (sdf_vals_tr[:, None] * sdf_grads_tr).sum(dim=0)

    # compute dF/dR
    # sdf normals
    sdf_normals = (R @ sdf_grads_tr.T).T  # Rotate SDF gradients
    sdf_normals = sdf_normals / sdf_normals.norm(dim=1, keepdim=True)

    # Compute cosine similarity
    drill_view_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    scene_normals_np = np.asarray(drill_view_pcd.normals)
    scene_normals = torch.tensor(scene_normals_np, dtype=torch.float32)
    cos_sim = (scene_normals * sdf_normals).sum(dim=1)  # Dot product

    outer = scene_normals.unsqueeze(2) * sdf_grads_tr.unsqueeze(1) 
    dF_dR = 2 * (1 - cos_sim).unsqueeze(1).unsqueeze(2) * outer 
    dF_dR = dF_dR.sum(dim=0)  # Sum over all points

    # Update translation t
    t = t - STEP_SIZE * dF_dt

    # Update Rotation R
    dR = -STEP_SIZE_ROT * dF_dR
    U, _, V = torch.svd(R + dR)
    R = U @ V.T

    current_loss = (sdf_vals_tr ** 2).mean().item() + (1 - cos_sim).mean().item()
    losses.append(current_loss)

    if plot_loss:
        line.set_ydata(losses)
        line.set_xdata(range(len(losses)))
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)  # Pause to update the plot

    updated_pcd.points = o3d.utility.Vector3dVector(t_pcd.numpy())
    vis.update_geometry(updated_pcd)
    vis.poll_events()
    vis.update_renderer()


vis.destroy_window()
plt.ioff()
plt.show()
