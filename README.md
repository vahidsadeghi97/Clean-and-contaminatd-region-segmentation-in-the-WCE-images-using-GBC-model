
This project aims to segment clear and contaminated regions in capsule endoscopy images by leveraging LAB color space transformation,3-D color histogram, K-means clustering, and multivariate Gaussian models. The approach needs few number of annotated images for the segmentation of clean and contaminated areas in the small bowel capsule endoscopy images.

Workflow Overview:
1. Data Preparation:
The number of 2000 capsule endoscopy images from the Kvasir capsule endoscopy dataset were converted from RGB to LAB color space.
A 3-D color histogram was computed for each image, with L, A, and B channels divided into 8 bins, resulting in a 512-dimensional feature vector for each image.
2. Clustering:
The K-means algorithm with K=20 was applied to the 512-dimensional feature vectors, clustering the images into 20 groups.
From each cluster, one image and its corresponding ground truth mask were randomly selected. These 20 images formed the model construction dataset.
3. Feature Extraction:
The color pixel intensity values of the clean regions in the selected images were concatenated together.
Similarly, the color pixel intensity values of the contaminated regions were concatenated to form a contaminated color pattern.
4. Gaussian Distribution Models:
The mean and covariance matrices of both clean and contaminated pixel intensity values were calculated, constructing two multivariate Gaussian distributions.
These distributions represent the clean and contaminated regions, respectively.
5. Testing Mode:
During testing, each pixel in a test image is evaluated by calculating its likelihood under the two Gaussian distributions (clean and contaminated).
Each pixel is assigned to the class (clean or contaminated) with the maximum probability.


Requirements: Install the required dependencies:

pip install -r requirements.txt

Run the model: Follow the instructions in the provided notebook or scripts to preprocess the data, train the Gaussian models, and perform testing on new images.

Features:
LAB color space transformation for better distinction between clean and contaminated regions.
K-means clustering to reduce the dataset and select representative images for training.
Gaussian distribution-based segmentation for pixel classification.
Applications:
This segmentation approach can aid in:

Improving diagnostic accuracy in capsule endoscopy by clearly identifying clean and contaminated areas.
Enhancing image preprocessing for further analysis.
License:
This project is licensed under the MIT License. See the LICENSE file for details.
