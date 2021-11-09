import numpy as np
import os
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Create the directory for output images
directory = './fig/'
if not os.path.isdir(directory):
    os.makedirs(directory)

# Fetch the datasets from openml
mnist = fetch_openml('mnist_784')

"""
Part1: PCA
Q2. ~ Q4.
"""
# Q2.
# Obtain the indices of all "5".
indices = np.where(mnist['target'].to_numpy() == '5')[0]
data_5 = mnist['data'].to_numpy()[indices]
mean_5 = np.mean(data_5, axis=0)
# Transpose the dataset to the form: X = [x1, x2, x3, ..., xN]
data_5_centered = (data_5 - mean_5).T
# Calculate the covariance matrix
covariance_mat = np.cov(data_5_centered)
# Calculate eigenvalue and eigenvector
eigen_val, eigen_vec = np.linalg.eig(covariance_mat)
# Eigenvector with each rows
eigen_vec = eigen_vec.T
# Sorting the eigenvector by its eigenvalue in descending order.
eigen_val_des = eigen_val[np.argsort(eigen_val)[::-1]]
eigen_vec_des = eigen_vec[np.argsort(eigen_val)[::-1]]
# Show the first 3 largest eigenvectors.
fig = plt.figure(figsize=(9, 3))
for i in range(3):
    plt.subplot(131 + i)
    plt.imshow((eigen_vec_des[i]).real.reshape(28, 28), 'gray')
    plt.title("λ = {0}".format(eigen_val_des[i].real))
    plt.axis('off')
plt.savefig(os.path.join(directory, 'Q2.png'))

# Q3.
origin_first_5 = data_5[0]
origin_first_5_mean = origin_first_5 - mean_5
images_5 = [origin_first_5]
bases = [3, 10, 30, 100]
fig = plt.figure(figsize=(15, 3))
plt.subplot(151)
plt.imshow(origin_first_5.reshape(28, 28), 'gray')
plt.title('Original 5')
plt.axis('off')
for i, n in enumerate(bases):
    plt.subplot(152 + i)
    # Obtain the first n largest eigenvectors and corresponding coefficients.
    eigen_vec_des_top_n = eigen_vec_des[:n]
    coef_5_top_n = np.dot(eigen_vec_des_top_n, origin_first_5_mean)
    # Reconstruct the signal by the eigenvectors.
    reconstruct_n_base = np.dot(eigen_vec_des_top_n.T, coef_5_top_n)
    plt.imshow((reconstruct_n_base + mean_5).real.reshape(28, 28), 'gray')
    plt.title("5 with {0}d".format(n))
    plt.axis('off')
plt.savefig(os.path.join(directory, 'Q3.png'))

# Q4.
first_10000_target = mnist['target'].to_numpy()[:10000]
first_10000_data = mnist['data'].to_numpy()[:10000]
# Obtain the indices of all "1", "3", "6".
indices = np.where((first_10000_target == '1') | (first_10000_target == '3') | (first_10000_target == '6'))[0]
data = first_10000_data[indices]
mean = np.mean(data, axis=0)
# Transpose the dataset to the form: X = [x1, x2, x3, ..., xN]
data_centered = (data - mean).T
# Calculate eigenvalue and eigenvector
covariance_mat = np.cov(data_centered)
# Calculate eigenvalue and eigenvector
eigen_val, eigen_vec = np.linalg.eig(covariance_mat)
# Eigenvector with each rows
eigen_vec = eigen_vec.T
# Sorting the eigenvector by its eigenvalue in descending order.
eigen_val_des = eigen_val[np.argsort(eigen_val)[::-1]]
eigen_vec_des = eigen_vec[np.argsort(eigen_val)[::-1]]
# Store the projection points (x, y) to the first 2 largest eigenvectors.
points_x = {'1': [], '3': [], '6': []}
points_y = {'1': [], '3': [], '6': []}
colors = {'1': 'r', '3': 'g', '6': 'b'}
for i, dc in enumerate(data_centered.T):
    eigen_vec_des_top_2 = eigen_vec_des[:2]
    coef_top_2 = np.dot(eigen_vec_des_top_2, dc)
    points_x[first_10000_target[indices][i]].append(coef_top_2[0].real)
    points_y[first_10000_target[indices][i]].append(coef_top_2[1].real)
fig = plt.figure(figsize=(6, 4))
for key,values in colors.items():
    plt.scatter(points_x[key], points_y[key], color=values, label=key)
plt.legend()
plt.savefig(os.path.join(directory, 'Q4.png'))

"""
Part2: OMP
Q5. ~ Q6.
"""
# Q5.
# Obtain training basis(pre-normalized)
basis = mnist['data'].to_numpy()[:10000]
length = np.linalg.norm(basis, axis=1)
basis = (basis.T / length).T
# Define the sparsity is 5.
sparsity = 5
origin_signal = mnist['data'].to_numpy()[10000]
# Bases and coefficients we choose.
sparse_basis, coefficients = [], []
# Residual sidgnal equals to original signal(Need deep copy)
residue_signal = origin_signal.copy()
for k in range(sparsity):
    max_product, max_base, max_index = 0, [], 0
    # Find the base vector that cause maximum product with residual signal.
    for i, base in enumerate(basis):
        product = abs(np.dot(base, residue_signal))
        if product > max_product:
            max_product, max_base, max_index = product, base, i
    # Calculate the coefficient vector by the pseudo inerse: (M^T*M)^(-1)M^T*x
    sparse_basis.append(max_base)
    M = np.array(sparse_basis).T
    pseudo_inverse = np.linalg.pinv(M.T @ M)
    coefficients = (pseudo_inverse @ M.T) @ origin_signal
    # Update residual signal.
    residue_signal = origin_signal - (M @ coefficients)
    # Delete the basis we choose.
    basis = np.delete(basis, max_index, axis=0)
# Show the first 5 largest bases we choose.
fig = plt.figure(figsize=(15, 3))
for i, base in enumerate(sparse_basis):
    plt.subplot(151 + i)
    plt.imshow(base.reshape(28, 28), 'gray')
    plt.title("Base {0}".format(i + 1))
    plt.axis('off')
plt.savefig(os.path.join(directory, 'Q5.png'))

# Q6.
origin_signal = mnist['data'].to_numpy()[10001]
# Define the sparsity is 5, 10, 40 and 200.
sparsity = [5, 10, 40, 200]
fig = plt.figure(figsize=(15, 3))
plt.subplot(151)
plt.imshow(origin_signal.reshape(28, 28), 'gray')
plt.title('L-2=0')
plt.axis('off')
for index, s in enumerate(sparsity):
    # Obtain training basis(pre-normalized)
    basis = mnist['data'].to_numpy()[:10000]
    length = np.linalg.norm(basis, axis=1)
    basis = (basis.T / length).T
    # Bases and coefficients we choose.
    sparse_basis, coefficients = [], []
    # Residual sidgnal equals to original signal(Need deep copy)
    residue_signal = origin_signal.copy()
    for k in range(s):
        max_product, max_base, max_index = 0, [], 0
        # Find the base vector that cause maximum product with residual signal.
        for i, base in enumerate(basis):
            product = abs(np.dot(base, residue_signal))
            if product > max_product:
                max_product, max_base, max_index = product, base, i
        # Calculate the coefficient vector by the pseudo inerse: (M^T*M)^(-1)M^T*x
        sparse_basis.append(max_base)
        M = np.array(sparse_basis).T
        pseudo_inverse = np.linalg.pinv(M.T @ M)
        coefficients = (pseudo_inverse @ M.T) @ origin_signal
        # Update residual signal.
        residue_signal = origin_signal - (M @ coefficients)
        # Delete the basis we choose.
        basis = np.delete(basis, max_index, axis=0)
    L2_norm = np.linalg.norm(residue_signal)
    plt.subplot(152 + index)
    plt.imshow((np.array(sparse_basis).T @ np.array(coefficients)).reshape(28, 28), 'gray')
    plt.title("L-2={0}".format(L2_norm))
    plt.axis('off')
plt.savefig(os.path.join(directory, 'Q6.png'))