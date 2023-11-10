import numpy as np
from lab_3.task_1 import create_gaussian_matrix


def normalize_matrix(matrix):
    return matrix / np.sum(matrix)


sizes = [3, 5, 7]
sigma = 1.0

print('task_2')
for size in sizes:
    gaussian_matrix = create_gaussian_matrix(size, sigma)
    normalized_matrix = normalize_matrix(gaussian_matrix)

    print(f"Normalized Gaussian Matrix (Size {size}x{size}, Sigma {sigma}):")
    print(normalized_matrix)
    print("\n")
