import open3d as o3d
import torch
import os
import pytorch_volumetric as pv
from pytorch_volumetric import sample_mesh_points
from pytorch_volumetric import voxel
from pytorch_volumetric import sdf
import numpy as np
import matplotlib.pyplot as plt


OBJS_DIR = "/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/objs/"
RESOLUTION = 0.005
DEVICE = "cpu"

obj = pv.MeshObjectFactory(os.path.join(OBJS_DIR, "drill.obj"))
drill_sdf = pv.MeshSDF(obj)

# query_range = np.array([
#     [-0.15, 0.2],
#     [0.01, 0.01],
#     [-0.1, 0.2],
# ])

# pv.draw_sdf_slice(drill_sdf, query_range, resolution=RESOLUTION, cmap="viridis")

# coords, pts = voxel.get_coordinates_and_points_in_grid(RESOLUTION, query_range, device=DEVICE)
# pts += torch.randn_like(pts) * 1e-6


pts = torch.rand(5, 3) * 3
sdf_vals, sdf_grads = drill_sdf(pts)

# print(pts.shape)
print(sdf_vals)
print()
print(sdf_grads)
