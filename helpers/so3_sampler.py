import torch
from scipy.spatial import Rotation
import numpy as np
import math

class SO3_Sampler():

    def __init__(self):
        self = self

    def random_so3_sample(n):
        """
        Sample SO(3) rotations using random quaternions.
        """

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

    def super_fibinacci_so3_samples(n_rotations):
        """
        Sample SO(3) rotations using the super Fibonacci method as proposed by M. Alexa (CVOR 2022)
        """

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
