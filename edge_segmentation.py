import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path

# Create output directory
output1_dir = "edge_detection_results"
output2_dir = "edge_detection_with_centroid_results"

os.makedirs(output1_dir, exist_ok=True)
os.makedirs(output2_dir, exist_ok=True)


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

# Start loop here
directory_path = Path('../../homing dataset/images_v2')

for file in directory_path.iterdir():
    # Load input image
    img = cv2.imread(str(file))
    (H, W) = img.shape[:2]

    # Create blob
    mean_pixel_values = np.average(img, axis=(0, 1))
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                                mean=(105, 117, 123),
                                swapRB=False, crop=False)
    
    net.setInput(blob)
    hed = net.forward()
    hed = hed[0, 0, :, :]
    hed = (255 * hed).astype("uint8")
    
    # Save HED result
    plt.figure()
    plt.imshow(hed, cmap='gray')
    # plt.title('HED Edge Detection')
    plt.axis('off')
    plt.savefig(f"{output1_dir}/{file.stem}_hed.png", bbox_inches='tight', dpi=150, pad_inches=0)
    plt.close()
    
    blur = cv2.GaussianBlur(hed, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    hed_color = cv2.cvtColor(hed, cv2.COLOR_GRAY2BGR)
    MIN_AREA = 50
    for i, centroid in enumerate(centroids[1:], start=1):
        area = stats[i, 4]
        if area > MIN_AREA:
            cv2.drawMarker(hed_color, (int(centroid[0]), int(centroid[1])),
                        color=(0, 0, 255), 
                        markerType=cv2.MARKER_CROSS, 
                        markerSize=20, 
                        thickness=2
                        )
    

    # Save HED with centroid result
    plt.figure()
    plt.imshow(cv2.cvtColor(hed_color, cv2.COLOR_BGR2RGB))
    # plt.title('HED Edge Detection')
    plt.axis('off')
    plt.savefig(f"{output2_dir}/{file.stem}_hed_c.png", bbox_inches='tight', dpi=150, pad_inches=0)
    plt.close()
    print(f"...finished with {file.stem}")