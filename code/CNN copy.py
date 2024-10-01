import tensorflow as tf
import keras
import scipy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define the CNN model
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    #Dropout(0.5),  # Helps with overfitting
    Dense(1, activation='relu')  # Binary classification (red or not red)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values
    shear_range=0.2,      # Shear augmentation
    zoom_range=0.2,       # Zoom augmentation
    horizontal_flip=True  # Horizontal flip augmentation
)

# Rescale the validation/test data
test_datagen = ImageDataGenerator(rescale=1./255)

print("training set generation...")
# Training data
train_set = train_datagen.flow_from_directory(
    'code/dataset/traffic_light_data/train/',       # Path to training set
    target_size=(28, 28),  # Resize images to 64x64
    batch_size=32,
    classes=['red', 'not red'],  # Include all three classes, we'll adjust labels later
    class_mode='binary',   # Binary classification
    shuffle=True
)

print("test dataset generation...")
# Test/validation data
test_set = test_datagen.flow_from_directory(
    'code/dataset/traffic_light_data/val/',        # Path to test set
    target_size=(28, 28),
    batch_size=32,
    classes=['red', 'not red'],
    class_mode='binary'
)

print("starting model training...")
# Train the model
history = model.fit(
    train_set,
    steps_per_epoch=train_set.samples // train_set.batch_size,
    epochs=20,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size
)

# Evaluate the model on the test set to calculate accuracy
test_loss, test_acc = model.evaluate(test_set, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save('model.keras')
