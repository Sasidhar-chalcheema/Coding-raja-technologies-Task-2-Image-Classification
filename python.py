import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import cv2
import os


def generate_dataset(num_samples_per_class, num_classes, img_height, img_width):
    X = []
    y = []

    for class_idx in range(num_classes):
        for _ in range(num_samples_per_class):

            img = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
            label = class_idx

            X.append(img)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y

num_samples_per_class = 100
num_classes = 5
img_height, img_width = 224, 224

X, y = generate_dataset(num_samples_per_class, num_classes, img_height, img_width)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(base_model.input, output)


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
