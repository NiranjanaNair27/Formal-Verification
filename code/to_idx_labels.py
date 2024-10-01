import numpy as np
import struct

# Define your labels
num_labels = 5  # Number of labels/images
label_class = 0  # Assuming all labels belong to the same class

# Create an array of labels (all the same class)
labels = np.array([label_class] * num_labels, dtype=np.uint8)

# IDX Header for labels
# Magic number for labels is 0x00000001
magic_number = 0x00000801
num_dims = 1  # Number of dimensions
labels_shape = labels.shape[0]  # Number of labels
print(labels_shape)

# Create the IDX file
with open('labels.idx', 'wb') as f:
    # Write the magic number
    f.write(struct.pack('>I', magic_number))
    # Write the number of dimensions
    f.write(struct.pack('>I', labels_shape))
    # Write the shape of the labels
    f.write(struct.pack('>I', num_dims))
    # Write the actual labels
    f.write(labels.tobytes())

print("Labels converted to labels.idx")
