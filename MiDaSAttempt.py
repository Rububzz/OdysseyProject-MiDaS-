import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# Download the MiDaS model from Intel-isl repository
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# OpenCV video capture setup
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Transform input for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

    # Normalize the output for better visualization
    normalized_output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
    depth_colormap = cv2.applyColorMap(normalized_output.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # Display the depth map using OpenCV
    cv2.imshow('Depth Map', depth_colormap)

    # Check for 'q' key to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()