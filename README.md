# Image Classification using PCA

This repository contains projects demonstrating the application of Principal Component Analysis (PCA) for image classification tasks. Two projects are included: digit recognition using the MNIST dataset and car vs. non-car image classification.

## Project 1: Digit Recognition with PCA

### Overview
- Utilized PCA to reduce the dimensionality of digit images from the MNIST dataset.
- Achieved 95% variance retention with 29 principal components, significantly reducing computational complexity.
- Implemented Logistic Regression achieving 97.2% accuracy on the test set.

### Key Steps and Technologies Used
- Loaded and preprocessed the MNIST dataset using scikit-learn and pandas.
- Applied PCA to extract essential features and reduce data dimensions.
- Trained a Logistic Regression model on the PCA-transformed data for digit classification.

### Results
- Successfully reduced the dimensionality of the dataset while maintaining high accuracy in digit recognition.
- Highlighted the effectiveness of PCA in feature extraction for machine learning tasks.

## Project 2: Car vs. Non-Car Image Classification with PCA

### Overview
- Implemented PCA to extract features from car and non-car images resized to 25x25 pixels.
- Used PCA subspace coefficients to reconstruct images and compared Mean Squared Error (MSE) for classification.

### Key Steps and Technologies Used
- Loaded and resized car and non-car images using Python Imaging Library (PIL).
- Computed covariance matrices and eigen decomposition to derive PCA components.
- Classified images based on reconstructed errors using PCA features.

### Results
- Successfully differentiated between car and non-car images based on PCA-derived features.
- Demonstrated PCA's effectiveness in image classification tasks with limited dimensional data.

## Repository Structure
- **`/project1/`**: Contains code and notebooks for digit recognition using PCA.
- **`/project2/`**: Contains code and notebooks for car vs. non-car image classification using PCA.
- **`/datasets/`**: Includes sample datasets used in the projects.
- **`/README.md`**: Markdown file providing an overview of the repository and project details.

## Requirements
- Python 3.x
- Required libraries: scikit-learn, pandas, numpy, PIL (Python Imaging Library)

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Image-classification-using-PCA
