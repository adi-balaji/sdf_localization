"""
This script writes a pcd file according to the obj mesh file with 700 points. It saves the direct obj point cloud, and a transformed point cloud. The ground truth transform is:

- Rotation: (pi/3, pi/6, 0) where the rotation matrix is
R = [[ 0.8660254  0.0,         0.5      ]
    [ 0.4330127  0.5       -0.75     ]
    [-0.25       0.8660254  0.4330127]]

- Translation: (0.35, 0.1, 0.05) 
"""

import numpy as np
import open3d as o3d
import os
import copy

def rotate_mesh(mesh, x, y, z):
    R = mesh.get_rotation_matrix_from_xyz((x, y, z))
    print(R)
    mesh.rotate(R, center=mesh.get_center())
    return mesh

def translate_mesh(mesh, x, y, z):
    mesh.translate([x, y, z])
    return mesh

OBJS_DIR = "/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/objs/"

mesh = o3d.io.read_triangle_mesh(os.path.join(OBJS_DIR, "drill.obj"))
mesh.compute_vertex_normals()

transformed_mesh = copy.deepcopy(mesh)
rotate_mesh(transformed_mesh, np.pi/3, np.pi/6, 0)
translate_mesh(transformed_mesh, 0.35, 0.1, 0.05)

pcd = mesh.sample_points_uniformly(number_of_points=700)
transformed_pcd = transformed_mesh.sample_points_uniformly(number_of_points=700)
o3d.visualization.draw_geometries([pcd, transformed_pcd])

o3d.io.write_point_cloud("/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/pcd/transformed_drill.pcd", transformed_pcd)

