import open3d as o3d
import torch
import os
import pytorch_volumetric as pv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import time

def sample_six_opposite_rotations():
    Rs = torch.zeros((6, 3, 3), dtype=torch.float32)

    Rs[0] = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
    Rs[1] = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32)
    Rs[2] = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)
    Rs[3] = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    Rs[4] = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    Rs[5] = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)

    return Rs

def calculate_slope(y_values):
    x_values = np.arange(len(y_values))
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    
    numerator = np.sum((x_values - x_mean) * (y_values - y_mean))
    denominator = np.sum((x_values - x_mean) ** 2)
    slope = numerator / denominator
    
    return slope

def has_converged(losses, window_size=50):

    if len(losses) < window_size:
        return False

    loss_window = losses[-window_size:]
    loss_window_slope = calculate_slope(loss_window)
    
    return np.abs(loss_window_slope) < 1e-7

def random_so3_sample(n):

    # Generate a random quaternion
    Rs = torch.zeros((n, 3, 3), dtype=torch.float32)

    for i in range(n):
        u1, u2, u3 = np.random.rand(3)  # Uniform random numbers in [0, 1)

        # Convert to quaternion
        q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        quaternion = np.array([q0, q1, q2, q3])

        # Convert quaternion to rotation matrix
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        Rs[i] = torch.tensor(rotation_matrix, dtype=torch.float32)
    return Rs

def super_fibinacci_so3_samples(n_rotations):

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

        R = Rotation.from_quat([q0, q1, q2, q3]).as_matrix()
        R = torch.tensor(R, dtype=torch.float32)

        R_samples[i] = R

    return R_samples

def pc_principal_axis(point_cloud):
    pc_np = np.asarray(point_cloud.points)
    pca_pc = PCA(n_components=3)
    pca_pc.fit(pc_np)
    eigen_vals_pc = pca_pc.explained_variance_
    axes_pc = pca_pc.components_
    return axes_pc[0] * 3 * np.sqrt(eigen_vals_pc[0]) 

def rot_mat_from_principal_axes(pcd_scene, pcd_gt):
    a = pc_principal_axis(pcd_scene)
    b = pc_principal_axis(pcd_gt)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    cos_theta = np.dot(a, b)
    sin_theta = np.linalg.norm(v)
    if sin_theta == 0:
        return np.eye(3)
    v = v / sin_theta
    v_cross = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    R = np.eye(3) + sin_theta * v_cross + (1 - cos_theta) * np.dot(v_cross, v_cross)

    R = torch.tensor(R, dtype=torch.float32)
    return R

OBJS_DIR = "objs/"
PCD_DIR = "pcd/"
RESOLUTION = 0.005
DEVICE = "cpu"
#1e-3 and 1e-4 is best
STEP_SIZE = 1e-3 # 1e-3, 1e-4
STEP_SIZE_ROT = 1e-4    # 1e-3, 1e-4
plot_loss = False
visualize = True
# R_samples = sample_six_opposite_rotations()
# R_samples = random_so3_sample(20)
# R_samples = super_fibinacci_so3_samples(20)

if visualize:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SDF Localization")

GT_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "drill.pcd"))
GT_pcd.paint_uniform_color([0.0, 0.0, 0.7])
updated_pcd = o3d.geometry.PointCloud()  # Initialize scene point cloud
if visualize:
    vis.add_geometry(GT_pcd)
    vis.add_geometry(updated_pcd)

obj = pv.MeshObjectFactory(os.path.join(OBJS_DIR, "drill.obj"))  # Create mesh object for PV
drill_sdf = pv.MeshSDF(obj)  # Compute SDF for the mesh object

drill_view_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "transformed_large_drill.pcd"))
drill_np = np.asarray(drill_view_pcd.points)
drill_pcd_tensor = torch.tensor(drill_np, dtype=torch.float32)  # Convert scene point cloud to tensor

min_loss = float('inf')
min_loss_R = None
min_loss_t = None

t = torch.zeros(3)
R = rot_mat_from_principal_axes(drill_view_pcd, GT_pcd).T
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
while len(losses) < 1000:
    
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
    dF_dR = dF_dR.sum(dim=0)

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

    if has_converged(losses):
        break

if visualize:
    vis.destroy_window()

if plot_loss:
    plt.ioff()
    plt.show()


print(f"\nTranslation (t): \n {-t} \n")
print(f"Rotation (R): \n {R.T}")