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
import json

class SDFLocalizer():

    def __init__(self):

        self.STEP_SIZE = 1e-3 # 1e-3, 1e-4
        self.STEP_SIZE_ROT = 1e-4    # 1e-3, 1e-4
        self.plot_loss = False
        self.visualize = True
        self.MAX_ITER = 1250
        self.GT_PATH = None
        self.gt_json_data = None

        self.sdf = None
        self.R = None
        self.t = None

        self.gt_pcd = None
        self.scene_pcd = None
        self.scene_pcd_tensor = None

        self.losses = []

    def __random_so3_sample(n):

        Rs = torch.zeros((n, 3, 3), dtype=torch.float32)

        for i in range(n):
            u1, u2, u3 = np.random.rand(3)

            q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
            q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
            q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
            q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
            quaternion = np.array([q0, q1, q2, q3])

            rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
            Rs[i] = torch.tensor(rotation_matrix, dtype=torch.float32)
        return Rs

    def __super_fibinacci_so3_samples(n_rotations):

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

    def __calculate_slope(self, y_values):
        x_values = np.arange(len(y_values))
        x_mean = np.mean(x_values)
        y_mean = np.mean(y_values)
        
        numerator = np.sum((x_values - x_mean) * (y_values - y_mean))
        denominator = np.sum((x_values - x_mean) ** 2)
        slope = numerator / denominator
    
        return slope

    def __has_converged(self, window_size=50):
        if len(self.losses) < window_size:
            return False

        loss_window = self.losses[-window_size:]
        loss_window_slope = self.__calculate_slope(loss_window)
        
        return np.abs(loss_window_slope) < 1e-7

    def __pc_principal_axis(self, point_cloud):
        pc_np = np.asarray(point_cloud.points)
        pca_pc = PCA(n_components=3)
        pca_pc.fit(pc_np)
        eigen_vals_pc = pca_pc.explained_variance_
        axes_pc = pca_pc.components_
        return axes_pc[0] * 3 * np.sqrt(eigen_vals_pc[0]) 

    def __rot_mat_from_principal_axes(self, pcd_scene, pcd_gt):
        a = self.__pc_principal_axis(pcd_scene)
        b = self.__pc_principal_axis(pcd_gt)

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
        
    def construct_sdf(self, obj_path):
        obj = pv.MeshObjectFactory(obj_path)  # Create mesh object for PV
        sdf = pv.MeshSDF(obj)  # Compute SDF for the mesh object
        self.sdf = sdf

    def load_gt_and_scene_pcd(self, gt_pcd, scene_pcd):
        self.gt_pcd = gt_pcd
        self.scene_pcd = scene_pcd
        self.scene_pcd_tensor = torch.tensor(np.array(self.scene_pcd.points), dtype=torch.float32)

    def load_gt_json(self, gt_json_path):
        self.GT_PATH = gt_json_path
        with open(gt_json_path, "r") as f:
            self.gt_json_data = json.load(f)

    def get_gt_Rt(self, id):
        Rt = self.gt_json_data[id]
        gt_R = torch.tensor(Rt["R"], dtype=torch.float32)
        gt_t = torch.tensor(Rt["t"], dtype=torch.float32)

        return gt_R, gt_t

    def initialize_Rt_with_principal_axes(self):
        self.R = self.__rot_mat_from_principal_axes(self.scene_pcd, self.gt_pcd).T
        # self.t = -torch.mean(self.scene_pcd_tensor, dim=0)
        self.t = torch.zeros(3, dtype=torch.float32)
    def localize(self):

        if self.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="SDF Localization")
            self.gt_pcd.paint_uniform_color([0.0, 0.0, 0.7])
            updated_pcd = o3d.geometry.PointCloud() 
            vis.add_geometry(self.gt_pcd)
            vis.add_geometry(updated_pcd)

        # Loss plotting setup
        if self.plot_loss:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            line, = ax.plot(self.losses, label="Loss")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title("Real-Time Loss Plot")
            ax.legend()

        # Objective function
        while len(self.losses) < self.MAX_ITER:
            
            t_pcd = self.scene_pcd_tensor @ self.R + self.t.T  # transformation
            sdf_vals_tr, sdf_grads_tr = self.sdf(t_pcd)  # SDF values and gradients

            # compute dF/dt
            dF_dt = 2 * (sdf_vals_tr[:, None] * sdf_grads_tr).sum(dim=0)

            # compute dF/dR
            # sdf normals
            sdf_normals = (self.R @ sdf_grads_tr.T).T  # Rotate SDF gradients
            sdf_normals = sdf_normals / sdf_normals.norm(dim=1, keepdim=True)

            # Compute cosine similarity
            self.scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            scene_normals_np = np.asarray(self.scene_pcd.normals)
            scene_normals = torch.tensor(scene_normals_np, dtype=torch.float32)
            cos_sim = (scene_normals * sdf_normals).sum(dim=1)  # Dot product

            outer = scene_normals.unsqueeze(2) * sdf_grads_tr.unsqueeze(1) 
            dF_dR = 2 * (1 - cos_sim).unsqueeze(1).unsqueeze(2) * outer 
            dF_dR = dF_dR.sum(dim=0)

            # Update translation t
            self.t = self.t - self.STEP_SIZE * dF_dt

            # Update Rotation R
            dR = -self.STEP_SIZE_ROT * dF_dR
            U, _, V = torch.svd(self.R + dR)
            self.R = U @ V.T

            current_loss = (sdf_vals_tr ** 2).mean().item() + (1 - cos_sim).mean().item()
            self.losses.append(current_loss)

            if self.plot_loss:
                line.set_ydata(self.losses)
                line.set_xdata(range(len(self.losses)))
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)  # Pause to update the plot

            updated_pcd.points = o3d.utility.Vector3dVector(t_pcd.numpy())

            if self.visualize:
                vis.update_geometry(updated_pcd)
                vis.poll_events()
                vis.update_renderer()

        if self.visualize:
            vis.destroy_window()

        if self.plot_loss:
            plt.ioff()
            plt.show()

        return self.R, self.t
    

if __name__ == "__main__":

    object_name = "mug"
    OBJS_DIR = "objs/"
    PCD_DIR = "pcd/"

    gt_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, f"{object_name}.pcd"))

    for i in range(25):

        scene_pcd = o3d.io.read_point_cloud(f"/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/test_pcds/{object_name}/{object_name}_{i}.pcd")

        sdfl = SDFLocalizer()
        sdfl.load_gt_and_scene_pcd(gt_pcd, scene_pcd)
        sdfl.load_gt_json(f"/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/test_pcds/{object_name}/ground_truths.json")
        sdfl.construct_sdf(os.path.join(OBJS_DIR, f"{object_name}.obj"))
        sdfl.initialize_Rt_with_principal_axes()
        R, t = sdfl.localize()


    