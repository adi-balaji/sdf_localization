import open3d as o3d
import torch
import os
import pytorch_volumetric as pv
from pytorch_volumetric import sample_mesh_points
from pytorch_volumetric import voxel
from pytorch_volumetric import sdf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time
from sklearn.decomposition import PCA

OBJS_DIR = "objs/"
PCD_DIR = "pcd/"

# Get numpy arrays of ground truth and transformed point cloud 
GT_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "drill.pcd"))
GT_np = np.asarray(GT_pcd.points)

drill_view_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "transformed_drill.pcd")) 
drill_np = np.asarray(drill_view_pcd.points)

# Principal component analysis on both point clouds in 3 dimensions 
pca_gt = PCA(n_components=3)
pca_transformed = PCA(n_components=3)

pca_gt.fit(GT_np)
pca_transformed.fit(drill_np)

#--- FIGURE 1: GROUND TRUTH --- 

# Plot point cloud 
fig_gt = plt.figure(1)
ax_gt = fig_gt.add_subplot(111, projection='3d')
ax_gt.scatter(GT_np[:, 0], GT_np[:, 1], GT_np[:, 2], alpha=0.2)

axes_gt = pca_gt.components_

# Plot principal component vector with highest eigenvalue 
# (by default prints all three axes but breaks after first)
for i, vec in enumerate(axes_gt):
    v = vec * 3 * np.sqrt(pca_gt.explained_variance_[i]) 
    ax_gt.quiver(pca_gt.mean_[0], pca_gt.mean_[1], pca_gt.mean_[2], v[0], v[1], v[2], color='r')
    break

ax_gt.set_xlabel('X')
ax_gt.set_ylabel('Y')
ax_gt.set_zlabel('Z')

#--- FIGURE 2: TRANSFORMED ---

# Plot point cloud 

fig_transformed = plt.figure(2)
ax_transformed = fig_transformed.add_subplot(111, projection='3d')
ax_transformed.scatter(drill_np[:, 0], drill_np[:, 1], drill_np[:, 2], alpha=0.2)

axes_transformed = pca_transformed.components_

# Plot principal component vector with highest eigenvalue
# (by default prints all three axes but breaks after first) 
for i, vector in enumerate(axes_transformed):
    v = vector * 3 * np.sqrt(pca_transformed.explained_variance_[i])
    ax_transformed.quiver(pca_transformed.mean_[0], pca_transformed.mean_[1], pca_transformed.mean_[2], v[0], v[1], v[2], color='r')
    break

ax_transformed.set_xlabel('X')
ax_transformed.set_ylabel('Y')
ax_transformed.set_zlabel('Z')


# Display plots
plt.show()