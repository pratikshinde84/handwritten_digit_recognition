# -*- coding: utf-8 -*-
"""tf_cnn.py — Fixed for TensorFlow 2.x"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Sequential

# ===============================
# Prepare Dataset
# ===============================
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

print("TRAIN IMAGES:", train_images.shape)
print("TEST IMAGES:", test_images.shape)

# ===============================
# Create Model
# ===============================
model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')   # ← use softmax for classification
])

# ===============================
# Compile Model
# ===============================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ===============================
# Train Model
# ===============================
epochs = 10
history = model.fit(train_images, train_labels, epochs=epochs)

# ===============================
# Visualize Training Results
# ===============================
acc = history.history['accuracy']
loss = history.history['loss']

plt.figure(figsize=(8, 6))
plt.plot(range(epochs), acc, label='Training Accuracy')
plt.plot(range(epochs), loss, label='Training Loss')
plt.legend(loc='lower right')
plt.title('Training Accuracy and Loss')
plt.show()

# ===============================
# Test Image Prediction
# ===============================
image = (train_images[1]).reshape(1, 28, 28, 1)
prediction = np.argmax(model.predict(image), axis=-1)
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {prediction[0]}")
print("Prediction of model:", prediction[0])

image = (train_images[2]).reshape(1, 28, 28, 1)
prediction = np.argmax(model.predict(image), axis=-1)
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {prediction[0]}")
print("Prediction of model:", prediction[0])

# ===============================
# Test Multiple Images
# ===============================
images = test_images[1:5]
print("Test images array shape:", images.shape)

plt.figure(figsize=(10, 4))
for i, test_image in enumerate(images, start=1):
    pred = np.argmax(model.predict(test_image.reshape(1, 28, 28, 1)), axis=-1)
    plt.subplot(1, 4, i)
    plt.axis('off')
    plt.title(f"Predicted: {pred[0]}")
    plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.show()

# ===============================
# Save Model
# ===============================
model.save("tf-cnn-model.h5")
print("[INFO] Model saved as tf-cnn-model.h5")

# ===============================
# Load Model and Test Again
# ===============================
loaded_model = models.load_model("tf-cnn-model.h5")
image = (train_images[2]).reshape(1, 28, 28, 1)
prediction = np.argmax(loaded_model.predict(image), axis=-1)
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Loaded model prediction: {prediction[0]}")
print("Loaded model prediction:", prediction[0])
plt.show()
