""" 

This program uses Tkinter Graphical User Interface (GUI) to select, and load the
weight file and image.

"""

print("Importing packages, please wait...")
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tkinter import Tk, filedialog


# 1. Select YOLO weight file
Tk().withdraw()
print("\nSelect the weight file")
model_path = filedialog.askopenfilename(
    title="Select the weight file (.pt)",
    filetypes=[("YOLO Weights", "*.pt")]
)
if not model_path:
    raise ValueError("No model selected!")

model = YOLO(model_path)
print(f"Loaded model: {os.path.basename(model_path)}")


# 2. Select an image file
print("\nSelect an image")
image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Images", "*.jpg *.jpeg *.png")]
)
if not image_path:
    raise ValueError("No image selected!")


# 3. Run inference
results = model.predict(source=image_path, save=False, verbose=False)
# Draw detections on the image
img_with_boxes = results[0].plot()
img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)


# 4. Display the image
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.axis("off")
plt.title(f"Detections from {os.path.basename(model_path)}")
plt.show()
