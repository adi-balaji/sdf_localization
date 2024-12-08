"""
https://marcalexa.github.io/superfibonacci/
"""

import numpy as np
import torch

def is_valid_rotation_matrix(R):
    Rt = torch.transpose(R, 0, 1)
    should_be_identity = torch.matmul(Rt, R)
    I = torch.eye(3, dtype=torch.float32)
    return torch.allclose(should_be_identity, I, atol=1e-6)

n = 50

phi = np.sqrt(2.0)
psi = 1.533751168755204288118041

Q = np.empty(shape=(n,4), dtype=float)

for i in range(n):
    s = i+0.5
    r = np.sqrt(s/n)
    R = np.sqrt(1.0-s/n)
    alpha = 2.0 * np.pi * s / phi
    beta = 2.0 * np.pi * s / psi
    Q[i,0] = r*np.sin(alpha)
    Q[i,1] = r*np.cos(alpha)
    Q[i,2] = R*np.sin(beta)
    Q[i,3] = R*np.cos(beta)

for q in Q:

    # convert quaternion to rotation matrix
    q0, q1, q2, q3 = q
    R = torch.tensor([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
    ], dtype=torch.float32)

    # Check if R is a valid rotation matrix
    # print(is_valid_rotation_matrix(R))

    print(q)


