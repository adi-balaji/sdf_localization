import open3d as o3d
import torch
import os
import pytorch_volumetric as pv
import math
import numpy as np
import matplotlib.pyplot as plt
import copy

def sample_n_rotations(n_rotations):

    R_samples = torch.zeros((n_rotations, 3, 3), dtype=torch.float32)
    phi = math.sqrt(2.0)
    psi = 1.533751168755204288118041

    for i in range(n_rotations):

        s = i+0.5
        r = math.sqrt(s/n_rotations)
        R = math.sqrt(1.0-s/n_rotations)
        alpha = 2.0 * torch.pi * s / phi
        beta = 2.0 * torch.pi * s / psi

        q = torch.tensor([r*math.sin(alpha), r*math.cos(alpha), R*math.sin(beta), R*math.cos(beta)], dtype=torch.float32)
        q0, q1, q2, q3 = q

        R = torch.tensor([
            [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
            [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
            [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
        ], dtype=torch.float32)

        R_samples[i] = R

    return R_samples

OBJS_DIR = "/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/objs/"
PCD_DIR = "/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/pcd/"
RESOLUTION = 0.005
DEVICE = "cpu"
STEP_SIZE = 1e-3 # 1e-3
STEP_SIZE_ROT = 1e-4    # 1e-3
plot_loss = False
visualize = True
R_samples = sample_n_rotations(20)

if visualize:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SDF Localization")

GT_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "bottle.pcd"))
GT_pcd.paint_uniform_color([0.0, 0.0, 0.7])
updated_pcd = o3d.geometry.PointCloud()  # Initialize scene point cloud
if visualize:
    vis.add_geometry(GT_pcd)
    vis.add_geometry(updated_pcd)

obj = pv.MeshObjectFactory(os.path.join(OBJS_DIR, "bottle.obj"))  # Create mesh object for PV
bottle_sdf = pv.MeshSDF(obj)  # Compute SDF for the mesh object

bottle_view_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "transformed_large_bottle.pcd"))
R_test = bottle_view_pcd.get_rotation_matrix_from_xyz((0.0, -0.0, 0.0))
bottle_view_pcd.rotate(R_test, center=bottle_view_pcd.get_center())

bottle_np = np.asarray(bottle_view_pcd.points)
bottle_pcd_tensor = torch.tensor(bottle_np, dtype=torch.float32)  # Convert scene point cloud to tensor

min_loss = float('inf')
min_loss_R = None
min_loss_t = None

for j, init_R in enumerate(R_samples):

    t = torch.mean(bottle_pcd_tensor, dim=0)
    R = init_R
    losses = []
    rot_losses = []

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
    for i in range(500):
        
        t_pcd = bottle_pcd_tensor @ R + t.T  # transformation
        sdf_vals_tr, sdf_grads_tr = bottle_sdf(t_pcd)  # SDF values and gradients

        # compute dF/dt
        dF_dt = 2 * (sdf_vals_tr[:, None] * sdf_grads_tr).sum(dim=0)

        # compute dF/dR
        # sdf normals
        sdf_normals = (R @ sdf_grads_tr.T).T  # Rotate SDF gradients
        sdf_normals = sdf_normals / sdf_normals.norm(dim=1, keepdim=True)

        # Compute cosine similarity
        bottle_view_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        scene_normals_np = np.asarray(bottle_view_pcd.normals)
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
        rot_losses.append((1 - cos_sim).mean().item())

        if plot_loss:
            line.set_ydata(losses)
            line.set_xdata(range(len(losses)))
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)  # Pause to update the plot

        updated_pcd.points = o3d.utility.Vector3dVector(t_pcd.numpy())

        if visualize:
            vis.update_geometry(updated_pcd)
            vis.poll_events()
            vis.update_renderer()

    print(f"Rot loss for trial {j}: {rot_losses[-1]}")

    if rot_losses[-1] < min_loss:
        min_loss = rot_losses[-1]
        min_loss_R = R
        min_loss_t = t

if visualize:
    vis.destroy_window()

if plot_loss:
    plt.ioff()
    plt.show()

print()
print("Rotation:")
print(min_loss_R)
print("Translation:")
print(min_loss_t)




