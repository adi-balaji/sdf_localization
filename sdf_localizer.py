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
import copy
from scipy.spatial import cKDTree

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class SDFLocalizer():

    def __init__(self):

        self.STEP_SIZE = 1e-4
        self.STEP_SIZE_ROT = 1e-5
        # (1e-3, 1e-6) is best for speed, accuracy and stability. (1e-4, 1e-5) is best for visualizing the process

        self.plot_loss = False
        self.visualize = False
        self.MAX_ITER = 500
        self.GT_PATH = None
        self.gt_json_data = None

        self.sdf = None
        self.R = torch.eye(3, dtype=torch.float32)
        self.t = torch.zeros(3)

        self.gt_pcd = None
        self.scene_pcd = None
        self.scene_pcd_tensor = None
        self.updated_pcd = o3d.geometry.PointCloud()

        self.losses = []
        self.final_translation_errors = 0.0
        self.final_add = 0.0

    def __pc_principal_axis(self, point_cloud):
        """
        Computes the principal axis of a point cloud using PCA from sklearn.decomposition

        Parameters: point_cloud (open3d.geometry.PointCloud): Point cloud to compute the principal axis for.
        Returns: np.ndarray: Principal axis of the point cloud (best axis according to eigenvalues from PCA)

        (not used)
        """
        pc_np = np.asarray(point_cloud.points)
        pca_pc = PCA(n_components=3)
        pca_pc.fit(pc_np)
        eigen_vals_pc = pca_pc.explained_variance_
        axes_pc = pca_pc.components_
        return axes_pc[0] * 3 * np.sqrt(eigen_vals_pc[0]) 
    
    def __pc_principal_two_axes(self, point_cloud):
        """
        Computes TWO principal axes of a point cloud using PCA from sklearn.decomposition

        Parameters: point_cloud (open3d.geometry.PointCloud): Point cloud to compute the principal axes for.
        Returns: np.ndarray (2, 3): 2 best principal axes of the point cloud (best axess according to eigenvalues from PCA)
        
        """        
        pc_np = np.asarray(point_cloud.points)
        pca_pc = PCA(n_components=2)
        pca_pc.fit(pc_np)
        eigen_vals_pc = pca_pc.explained_variance_
        axes_pc = pca_pc.components_
        return axes_pc

    def __rot_mat_from_principal_axes(self, pcd_scene, pcd_gt):
        """
        Computes the rotation matrix to align the principal axis of the scene to the SDF points.

        (not used)
        """
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
    
    def __compute_rotation_matrix_from_two_axes(self, scene_pcd, gt_pcd):
        """
        Computes the rotation matrix to align two principal axes of the scene to the SDF goal.

        Parameters:
            scene_axes (np.ndarray): A (2, 3) numpy array representing two principal axes of the scene point cloud.
            sdf_axes (np.ndarray): A (2, 3) numpy array representing two principal axes of the SDF goal point cloud.

        Returns:
            torch.tensor: R (3, 3) rotation matrix that aligns the scene axes to the SDF axes.
        """

        scene_axes = self.__pc_principal_two_axes(scene_pcd)
        sdf_axes = self.__pc_principal_two_axes(gt_pcd)

        scene_axes = np.asarray(scene_axes)
        sdf_axes = np.asarray(sdf_axes)
        
        scene_third_axis = np.cross(scene_axes[0], scene_axes[1])
        sdf_third_axis = np.cross(sdf_axes[0], sdf_axes[1])
        
        # normalize the third axis
        scene_third_axis = scene_third_axis / np.linalg.norm(scene_third_axis)
        sdf_third_axis = sdf_third_axis / np.linalg.norm(sdf_third_axis)
        
        # construct 3 axis frame
        scene_full_axes = np.vstack((scene_axes, scene_third_axis)).T  # (3, 3)
        sdf_full_axes = np.vstack((sdf_axes, sdf_third_axis)).T        # (3, 3)
        
        # compute the rotation matrix by aligning the 3 axis frames
        R = sdf_full_axes @ np.linalg.inv(scene_full_axes)
        
        # enforce SO(3) constraint using SVD
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        return torch.tensor(R, dtype=torch.float32)
        
    def construct_sdf(self, obj_path):
        """
        Constructs the SDF for the object mesh using pytorch volumetric.
        
        Parameters: obj_path (str): Path to the object mesh file.
        """


        obj = pv.MeshObjectFactory(obj_path)  # Create mesh object for PV
        sdf = pv.MeshSDF(obj)  # Compute SDF for the mesh object
        self.sdf = sdf

    def load_gt_and_scene_pcd(self, gt_pcd, scene_pcd):
        """
        Loads the ground truth and scene point clouds into SDFLocalizer.
        """

        self.gt_pcd = gt_pcd
        self.scene_pcd = scene_pcd
        self.scene_pcd_tensor = torch.tensor(np.array(self.scene_pcd.points), dtype=torch.float32)

    def load_gt_json(self, gt_json_path):
        """
        Loads the ground truths JSON file into SDFLocalizer.
        """

        self.GT_PATH = gt_json_path
        with open(gt_json_path, "r") as f:
            self.gt_json_data = json.load(f)

    def get_gt_Rt(self, id):
        """
        Returns the ground truth rotation and translation for a given pcd ID.
        """

        Rt = self.gt_json_data[id]
        gt_R = torch.tensor(Rt["R"], dtype=torch.float32)
        gt_t = torch.tensor(Rt["t"], dtype=torch.float32)

        return gt_R, gt_t

    def initialize_Rt_with_principal_axes(self):
        """
        Initializes the R and t for localization
        """

        self.R = self.__compute_rotation_matrix_from_two_axes(self.scene_pcd, self.gt_pcd).T # initialize rotation with 2 principal axes
        self.t = -torch.mean(self.scene_pcd_tensor, dim=0) # initialize translation to the center of the scene point cloud

    def localize(self): 
        """
        Performs SDF guided object localization

        Returns: R (torch.tensor): Estimated rotation matrix
                t (torch.tensor): Estimated translation vector
        """

        if self.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="SDF Localization")
            self.gt_pcd.paint_uniform_color([0.0, 0.0, 0.7])
            vis.add_geometry(self.gt_pcd)
            vis.add_geometry(self.updated_pcd)

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

            # compute cosine similarity
            self.scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            scene_normals_np = np.asarray(self.scene_pcd.normals)
            scene_normals = torch.tensor(scene_normals_np, dtype=torch.float32)
            cos_sim = (scene_normals * sdf_normals).sum(dim=1)  # Dot product

            outer = scene_normals.unsqueeze(2) * sdf_grads_tr.unsqueeze(1) 
            dF_dR = 2 * (1 - cos_sim).unsqueeze(1).unsqueeze(2) * outer 
            dF_dR = dF_dR.sum(dim=0)

            # update translation t
            self.t = self.t - self.STEP_SIZE * dF_dt

            # update Rotation R
            dR = -self.STEP_SIZE_ROT * dF_dR
            U, _, V = torch.svd(self.R + dR)
            self.R = U @ V.T

            current_loss = (sdf_vals_tr ** 2).mean().item() + (1 - cos_sim).mean().item()
            self.losses.append(current_loss)

            self.updated_pcd.points = o3d.utility.Vector3dVector(t_pcd.numpy())

            if self.plot_loss:
                line.set_ydata(self.losses)
                line.set_xdata(range(len(self.losses)))
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)  # Pause to update the plot

            if self.visualize:
                
                vis.update_geometry(self.updated_pcd)
                vis.poll_events()
                vis.update_renderer()

        self.final_translation_error = torch.norm(torch.tensor(self.gt_pcd.get_center() - self.updated_pcd.get_center())) # L2 norm translation error
        
        kdtree = cKDTree(np.array(self.updated_pcd.points))  # Build KDTree for the estimated point cloud
        distances, _ = kdtree.query(np.array(self.gt_pcd.points))  # Find nearest neighbors for each GT point
        self.final_add = torch.mean(torch.tensor(distances))

        if self.visualize:
            vis.destroy_window()

        if self.plot_loss:
            plt.ioff()
            plt.show()

        return self.R, self.t
    

if __name__ == "__main__":
    """
    To test out the SDF localizer on the drill, ensure you unzip the test_pcds.zip file in the root directory of the repo.
    Simply run sdf_localizer.py to test the localizer on the drill object.

    Change file paths as necessary.

    Ensure the essential libraries are installed:
    - open3d
    - pytorch_volumetric
    - scipy
    - numpy
    - matplotlib
    - sklearn
    - pytorch

    """

    OBJS_DIR = "objs/"
    PCD_DIR = "pcd/"
    TEST_RESULT_DIR = "test_result_files/Rt_error_add_test_result.json"
    object_name = "hammer" # try different objects, check test_pcds folder for options

    with open(TEST_RESULT_DIR, "r") as f:
        test_results = json.load(f)

    R_errs = []
    t_errs = []
    ADD_errs = []

    for i in range(25): #do not change, there are only 25 test files per object

        gt_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, f"{object_name}.pcd"))
        scene_pcd = o3d.io.read_point_cloud(f"test_pcds/{object_name}/{object_name}_{i}.pcd")

        sdfl = SDFLocalizer()
        sdfl.visualize = True #visualize localization in open3d
        sdfl.load_gt_and_scene_pcd(gt_pcd, scene_pcd) #load gt and scene pcds
        sdfl.load_gt_json(f"test_pcds/{object_name}/ground_truths.json") #load gt json file
        sdfl.construct_sdf(os.path.join(OBJS_DIR, f"{object_name}.obj")) #construct SDF using pytorch volumetric
        sdfl.initialize_Rt_with_principal_axes() #initialize R and t with principal axes
        R, t = sdfl.localize() #run localization and get R, t

        #calculate metrics
        gt_R, gt_t = sdfl.get_gt_Rt(f"{object_name}_{i}")
        model_points = copy.deepcopy(sdfl.scene_pcd)
        model_points.translate(gt_t).rotate(gt_R)
        result_points = copy.deepcopy(sdfl.scene_pcd)
        result_points.translate(t).rotate(R)

        rotation_error = torch.norm(torch.eye(3) - R @ gt_R.T).item() #deviation of matrix products from identity
        translation_error = sdfl.final_translation_error.item() # L2 norm
        ADD = sdfl.final_add.item()

        print(f"R, t, ADD error: {(rotation_error, translation_error, ADD)}")

        R_errs.append(rotation_error)
        t_errs.append(translation_error)
        ADD_errs.append(ADD)

    test_results[f"{object_name}"] = {"R_err" : R_errs, "t_err" : t_errs, "ADD_err" : ADD_errs}
    with open(TEST_RESULT_DIR, "w") as f:
        json.dump(test_results, f, indent=4)


        




    