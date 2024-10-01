import os
import random
from PIL import Image
import numpy as np
import struct

# Define the directory with images and the number of images to convert
image_dir = "code/dataset/traffic_light_data/train/red/"  # Replace with your image directory
num_images_to_convert = 5     # Change this to how many images you want to process
output_file = "output.idx"    # Output .idx file name

# Function to write image data to .idx format
def save_to_idx(images, labels, output_file):
    # Define .idx file headers for images and labels
    num_images = len(images)
    rows, cols = images[0].shape[0], images[0].shape[1]

    with open(output_file, 'wb') as f:
        # Write magic number and metadata for images (3-dimensional)
        f.write(struct.pack('>IIII', 0x00000E03, num_images, rows, cols))
        for image in images:
            # Flatten each image and write pixel data
            f.write(image.astype(np.float32).tobytes())


# Get a list of all image filenames in the directory
all_images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Randomly select a set of images
selected_images = random.sample(all_images, num_images_to_convert)

# Arrays to hold the image data and dummy labels (labels can be modified as per your use case)
image_data = []
dummy_labels = []

# Loop through the selected images and convert them
for image_name in selected_images:
    image_path = os.path.join(image_dir, image_name)
    
    # Open the image and convert to NumPy array (preserving color)
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Ensure all images are the same size (resize if necessary)
    if image_array.shape != (64, 64, 3):  # Example: (28, 28, 3) is for RGB images of size 28x28
        image = image.resize((64, 64))  # Resize if necessary
        image_array = np.array(image)

    image_array = image_array / 255.0
    #print(image_array)

    # Append the processed image data to the list
    image_data.append(image_array)
    dummy_labels.append(0)  # You can adjust labels as needed for classification

# Convert image data to .idx format
save_to_idx(image_data, dummy_labels, output_file)

print(f"Converted {num_images_to_convert} images to {output_file}")

