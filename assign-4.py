from PIL import Image
import numpy as np
#loading images and converting into arrays
def loadresizeImage(image_path, targetsize=(25,25)):
    image = Image.open(image_path)
    resizedimage = image.resize(targetsize)
    return resizedimage

carimages = []
for i in range(1, 11):
    carimage = loadresizeImage(f'car{i}.jpg')
    carimagearray = np.array(carimage)
    carimages.append(carimagearray.flatten())

carimagesmatrix = np.stack(carimages)

noncarimages = []
for i in range(1, 11):
    noncarimage = loadresizeImage(f'noncar{i}.jpg')
    noncarimagearray = np.array(noncarimage)
    noncarimages.append(noncarimagearray.flatten())

noncarimagesmatrix = np.stack(noncarimages)

### 1 ###
#1a) computing avg of car and non car images
car_mean = np.mean(carimagesmatrix, axis=0)
noncar_mean = np.mean(noncarimagesmatrix, axis=0)
#1b) computing covariance matrices
car_covariance = np.cov(carimagesmatrix.T)
noncar_covariance = np.cov(noncarimagesmatrix.T)
#computing eigen values and eigen vectors
car_eigenvalues, car_eigenvectors = np.linalg.eigh(car_covariance)
noncar_eigenvalues, noncar_eigenvectors = np.linalg.eigh(noncar_covariance)
#computing PCA subspace by taking 9 eigen vectors corresponding to 9 highest eigen values
car_eigenvectors_subspace = car_eigenvectors[:, -9:]
noncar_eigenvectors_subspace = noncar_eigenvectors[:, -9:]
#1e) PCA subspace coeff for each training images
car_subspace_coeff = np.dot((carimagesmatrix-car_mean),car_eigenvectors_subspace)
noncar_subspace_coeff = np.dot((noncarimagesmatrix-noncar_mean),noncar_eigenvectors_subspace)

### 2 ###
car_car = 0
car_noncar = 0
noncar_car = 0
noncar_noncar = 0

# Classification for car images
for i in range(10):
    if i == 9:
        j = 0
    else:
        j = i + 1

    car_subspace = car_subspace_coeff[i:j, :]

    reconstructed_car1 = np.dot(car_subspace, car_eigenvectors_subspace.T) + car_mean
    reconstructed_car2 = np.dot(car_subspace, noncar_eigenvectors_subspace.T) + noncar_mean

    mse1 = np.mean((carimagesmatrix[i:j, :] - reconstructed_car1) ** 2)
    mse2 = np.mean((carimagesmatrix[i:j, :] - reconstructed_car2) ** 2)

    if mse1 < mse2:
        car_car += 1
    else:
        car_noncar += 1

# Classification for non-car images
for i in range(10):
    if i == 9:
        j = 0
    else:
        j = i + 1

    noncar_subspace = noncar_subspace_coeff[i:j, :]

    reconstructed_noncar1 = np.dot(noncar_subspace, car_eigenvectors_subspace.T) + car_mean
    reconstructed_noncar2 = np.dot(noncar_subspace, noncar_eigenvectors_subspace.T) + noncar_mean

    mse1 = np.mean((noncarimagesmatrix[i:j, :] - reconstructed_noncar1) ** 2)
    mse2 = np.mean((noncarimagesmatrix[i:j, :] - reconstructed_noncar2) ** 2)

    if mse1 < mse2:
        noncar_car += 1
    else:
        noncar_noncar += 1

print("CAR classified as CAR:", car_car)
print("CAR classified as NOT CAR:", car_noncar)
print("NOT CAR classified as CAR:", noncar_car)
print("NOT CAR classified as NOT CAR:", noncar_noncar)
