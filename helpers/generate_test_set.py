import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation
import copy
import json

def random_so3_sample(n):

    # Generate a random quaternion
    Rs = np.zeros((n, 3, 3))

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
        Rs[i] = rotation_matrix
    return Rs

def random_t_sample(n):
    return np.random.rand(n, 3)

def rotate_mesh(mesh, x, y, z):
    R = mesh.get_rotation_matrix_from_xyz((x, y, z))
    print(R)
    mesh.rotate(R, center=mesh.get_center())
    return mesh

def translate_mesh(mesh, x, y, z):
    mesh.translate([x, y, z])
    return mesh

object_name = "drill"
num_samples = 25
OBJS_DIR = "/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/objs/"
WRITE_DIR = f"/Users/adibalaji/Desktop/UMICH-24-25/manip/sdf_localization/test_pcds/{object_name}"
gt_log_path = os.path.join(WRITE_DIR, "ground_truths.json")
R_samples = random_so3_sample(num_samples)
t_samples = random_t_sample(num_samples)

mesh = o3d.io.read_triangle_mesh(os.path.join(OBJS_DIR, f"{object_name}.obj"))
mesh.compute_vertex_normals()
gt_log = {}

for i, R_sample in enumerate(R_samples):
    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.rotate(R_sample, center=transformed_mesh.get_center())
    transformed_mesh.translate(t_samples[i])
    pcd = transformed_mesh.sample_points_uniformly(number_of_points=500)

    o3d.io.write_point_cloud(f"{WRITE_DIR}/{object_name}_{i}.pcd", pcd)
    gt_log[f"{object_name}_{i}"] = {
        "R": R_sample.tolist(),
        "t": t_samples[i].tolist()
    }

with open(gt_log_path, "w") as f:
    json.dump(gt_log, f, indent=4)
