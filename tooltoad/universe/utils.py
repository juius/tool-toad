import numpy as np


# Function to create uniformly distributed points on a sphere using the Fibonacci lattice
def fibonacci_sphere(samples: int, radius: float):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append(np.array([x, y, z]) * radius)
    return np.array(points)


# Function to generate a random rotation matrix
def get_rotation_matrix(angles=np.ndarray) -> np.ndarray:
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    Ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    Rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    return Rz @ Ry @ Rx
