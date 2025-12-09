import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = "edge_detection_results"
os.makedirs(output_dir, exist_ok=True)


class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


# Load pre-trained model
protoPath = "hed_model/deploy.prototxt"
modelPath = "hed_model/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Register crop layer
cv2.dnn_registerLayer("Crop", CropLayer)

# Load input image
img = cv2.imread("0036_destination.png")
(H, W) = img.shape[:2]

# Save original image
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.savefig(f"{output_dir}/01_original.png", bbox_inches='tight', dpi=150)
plt.close()

# Create blob
mean_pixel_values = np.average(img, axis=(0, 1))
blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                             mean=(105, 117, 123),
                             swapRB=False, crop=False)

# Save blob visualization
blob_for_plot = np.moveaxis(blob[0, :, :, :], 0, 2).astype(np.float32)
plt.figure()
plt.imshow(blob_for_plot)
plt.title('Blob (Preprocessed)')
plt.savefig(f"{output_dir}/02_blob.png", bbox_inches='tight', dpi=150)
plt.close()

# Perform edge detection
net.setInput(blob)
hed = net.forward()
hed = hed[0, 0, :, :]
hed = (255 * hed).astype("uint8")

# Save HED result
plt.figure()
plt.imshow(hed, cmap='gray')
plt.title('HED Edge Detection')
plt.savefig(f"{output_dir}/03_hed_edges.png", bbox_inches='tight', dpi=150)
plt.close()

# Connected component labeling
blur = cv2.GaussianBlur(hed, (3, 3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Save threshold result
plt.figure()
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded')
plt.savefig(f"{output_dir}/04_threshold.png", bbox_inches='tight', dpi=150)
plt.close()

# Perform connected component labeling
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)

# Create false color image
colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]
false_colors = colors[labels]

# Save false colors
plt.figure()
plt.imshow(false_colors)
plt.title('Connected Components (False Colors)')
plt.savefig(f"{output_dir}/05_false_colors.png", bbox_inches='tight', dpi=150)
plt.close()

# Draw centroids
false_colors_centroid = false_colors.copy()
for centroid in centroids:
    cv2.drawMarker(false_colors_centroid, (int(centroid[0]), int(centroid[1])),
                   color=(255, 255, 255), markerType=cv2.MARKER_CROSS)

# Save with centroids
plt.figure()
plt.imshow(false_colors_centroid)
plt.title('With Centroids')
plt.savefig(f"{output_dir}/06_centroids.png", bbox_inches='tight', dpi=150)
plt.close()

# Filter by area
MIN_AREA = 50
false_colors_area_filtered = false_colors.copy()
for i, centroid in enumerate(centroids[1:], start=1):
    area = stats[i, 4]
    if area > MIN_AREA:
        cv2.drawMarker(false_colors_area_filtered, (int(centroid[0]), int(centroid[1])),
                       color=(255, 255, 255), markerType=cv2.MARKER_CROSS)

# Save filtered result
plt.figure()
plt.imshow(false_colors_area_filtered)
plt.title('Area Filtered (>50 pixels)')
plt.savefig(f"{output_dir}/07_area_filtered.png", bbox_inches='tight', dpi=150)
plt.close()

# Save composite figure
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(222)
plt.imshow(hed, cmap='gray')
plt.title('HED Edge Detection')
plt.subplot(223)
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded')
plt.subplot(224)
plt.imshow(false_colors_area_filtered)
plt.title('Segmented Objects')
plt.tight_layout()
plt.savefig(f"{output_dir}/08_composite.png", bbox_inches='tight', dpi=150)
plt.close()

print(f"All images saved to '{output_dir}/' directory")

# Extract properties using regionprops
from skimage import measure
props = measure.regionprops_table(labels, intensity_image=img,
                                  properties=['label', 'area', 'equivalent_diameter',
                                            'mean_intensity', 'solidity'])

import pandas as pd
df = pd.DataFrame(props)
df = df[(df.area > 50) & (df.area < 10000)]
print("\nObject properties:")
print(df.head())

# Save dataframe to CSV
df.to_csv(f"{output_dir}/object_properties.csv", index=False)
print(f"\nObject properties saved to '{output_dir}/object_properties.csv'")