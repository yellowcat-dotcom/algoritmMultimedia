import numpy as np


def gaussian(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)


def create_gaussian_matrix(size, sigma):
    center = size // 2
    gaussian_matrix = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            gaussian_matrix[i, j] = gaussian(x, y, sigma)

    return gaussian_matrix


def main():
    sizes = [3, 5, 7]
    sigma = 1.0  # Cреднее квадратичное отклонение.

    for size in sizes:
        gaussian_matrix = create_gaussian_matrix(size, sigma)
        print(f"Gaussian Matrix (Size {size}x{size}, Sigma {sigma}):")
        print(gaussian_matrix)
        print("\n")

if __name__ == '__main__':
    main()