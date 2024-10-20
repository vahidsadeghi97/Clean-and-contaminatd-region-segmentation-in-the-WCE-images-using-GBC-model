# Importing libraries
from time import time
start=time()
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2
from glob import glob
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr
capsule_mask=cv2.imread(r'E:\phd thesis articles\cleanliness\Plos One\data\Kvasir capsule endoscopy dataset\mask_Kvasir.jpg')[:,:,0]
# Function to convert RGB to LAB and calculate 3D histogram
def compute_lab_histogram(image, bins=(8, 8, 8)):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    hist = cv2.calcHist([lab_image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


# Function to calculate Dice Coefficient (F1-score)
def dice_coeff(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


# Function to calculate Intersection over Union (IoU)
def IntOUnion(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union if union != 0 else 0


# Function to calculate accuracy, precision, recall, and specificity
def measures(gt, seg):
    tp = np.sum((gt == 1) & (seg == 1))
    tn = np.sum((gt == 0) & (seg == 0))
    fp = np.sum((gt == 0) & (seg == 1))
    fn = np.sum((gt == 1) & (seg == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    accuracy = (tp + tn) / gt.size

    return accuracy, precision, recall, specificity


# Function to compute multivariate normal distribution PDF
def multivariate_normal(x, mean, covariance):
    d = mean.shape[0]
    x_c = x - mean
    covariance_inv = np.linalg.inv(covariance)
    normalization_factor = 1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance)))
    exponent = -0.5 * (x_c.T @ covariance_inv @ x_c)
    return normalization_factor * np.exp(exponent)


# Load images and masks
img_path = sorted(glob(r'E:\phd thesis articles\cleanliness\Plos One\data\Kvasir capsule endoscopy dataset\imgs\*.*'))
mask_path = sorted(glob(r'E:\phd thesis articles\cleanliness\Plos One\data\Kvasir capsule endoscopy dataset\masks\*.*'))

# Compute LAB histogram for each image
image_histograms, image_names = [], []
for image_file in img_path:
    image = cv2.imread(image_file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = compute_lab_histogram(image_rgb)
    image_histograms.append(hist)
    image_names.append(os.path.basename(image_file).split('.')[0])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=20, random_state=42)
kmeans.fit(image_histograms)
labels = kmeans.labels_

# Select one random image from each cluster
selected_images = []
for cluster in range(20):
    cluster_indices = np.where(labels == cluster)[0]
    if len(cluster_indices) > 0:
        random_index = random.choice(cluster_indices)
        selected_images.append(image_names[random_index])

# Extract pixel values for the selected images
B1, G1, R1, B2, G2, R2 = [], [], [], [], [], []
for img_name in selected_images:
    img_index = image_names.index(img_name)
    img = cv2.imread(img_path[img_index])
    mask = cv2.imread(mask_path[img_index])

    b, g, r = cv2.split(img)
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if mask[m, n, 0] == 0:
                B1.append(b[m, n])
                G1.append(g[m, n])
                R1.append(r[m, n])
            else:
                if (b[m, n] != 0 and g[m, n] != 0 and r[m, n] != 0):
                    B2.append(b[m, n])
                    G2.append(g[m, n])
                    R2.append(r[m, n])

# Compute mean and covariance for contaminated and clean regions
mean1 = np.array([np.mean(B1), np.mean(G1), np.mean(R1)])
cov1 = np.cov(np.array([B1, G1, R1]).reshape(3, -1))
mean2 = np.array([np.mean(B2), np.mean(G2), np.mean(R2)])
cov2 = np.cov(np.array([B2, G2, R2]).reshape(3, -1))

# Initialize metrics lists
GT, PRED, DSC, IOU, ACC, PREC, REC, SPEC, area_under_roc = [], [], [], [], [], [], [], [], []

# Create a set of selected image names for fast lookup
selected_image_set = set(selected_images)
end=time()
print('GBC model construction time: ',end-start)
# Load test images and masks
test_img_path = sorted(glob(r'E:\phd thesis articles\cleanliness\Plos One\data\Kvasir capsule endoscopy dataset\imgs\*.*'))
test_mask_path = sorted(glob(r'E:\phd thesis articles\cleanliness\Plos One\data\Kvasir capsule endoscopy dataset\masks\*.*'))

# Testing on new images
for j in range(len(test_img_path)):
    img_name = os.path.basename(test_img_path[j]).split('.')[0]

    # Skip testing on selected images
    if img_name in selected_image_set:
        continue

    print(f'Processing image number: {j}')
    test_img = cv2.imread(test_img_path[j])
    gt_mask = cv2.imread(test_mask_path[j])
    pdf = np.zeros((test_img.shape[0], test_img.shape[1]), dtype='uint8')
    probability_map = np.zeros((test_img.shape[0], test_img.shape[1]), dtype='float16')

    inf_region_gt, inf_region_pred = 0, 0
    for k in range(test_img.shape[0]):
        for l in range(test_img.shape[1]):
            p_contaminated = (len(R2) / (len(R2) + len(R1))) * multivariate_normal(test_img[k, l, :], mean1, cov1)
            p_clean = (len(R1) / (len(R2) + len(R1))) * multivariate_normal(test_img[k, l, :], mean2, cov2)
            probability_map[k, l] = p_clean / (p_contaminated + p_clean) if (p_contaminated + p_clean) > 0 else 0

            if p_contaminated < p_clean:
                pdf[k, l] = 255
                inf_region_pred += 1
            if gt_mask[k, l, 0] == 255:
                inf_region_gt += 1
    for m in range(test_img.shape[0]):
        for n in range(test_img.shape[1]):
            if capsule_mask[m,n]==0:
                pdf[m,n]=255
    GT.append(inf_region_gt / (pdf.size))
    PRED.append(inf_region_pred / (pdf.size))

    gt_mask = (gt_mask[:, :, 0] // 255).astype('bool')  # Convert mask to binary
    seg = (pdf.astype('bool'))
    ground_truth_labels = gt_mask.flatten()
    predicted_probabilities = probability_map.flatten()
    # Calculate evaluation metrics
    if np.unique(ground_truth_labels).size > 1:  # Check if there are both classes
        fpr, tpr, _ = roc_curve(ground_truth_labels, predicted_probabilities)
        auroc = auc(fpr, tpr)
    else:
        auroc = np.nan  # Set to NaN if only one class is present

    DSC.append(dice_coeff(gt_mask, seg))
    IOU.append(IntOUnion(gt_mask, seg))
    accuracy, precision, recall, specificity = measures(gt_mask, seg)

    ACC.append(accuracy)
    PREC.append(precision)
    REC.append(recall)
    SPEC.append(specificity)
    area_under_roc.append(auroc)

# Print summary statistics for metrics
print("Mean accuracy: ", np.mean(ACC))
print("STD accuracy: ", np.std(ACC))
print("Mean precision: ", np.mean(PREC))
print("STD precision: ", np.std(PREC))
print("Mean recall: ", np.mean(REC))
print("STD recall: ", np.std(REC))
print("Mean IoU: ", np.mean(IOU))
print("STD IoU: ", np.std(IOU))
print("Mean DSC: ", np.mean(DSC))
print("STD DSC: ", np.std(DSC))
print("Mean specificity: ", np.mean(SPEC))
print("STD specificity: ", np.std(SPEC))
print("Mean AUROC: ", np.nanmean(area_under_roc))
print("STD AUROC: ", np.nanstd(area_under_roc))
print("Pearson correlation coefficient between the gastroenterologist and GBC model: ", pearsonr(GT, PRED)[0])
print('Finished processing.')