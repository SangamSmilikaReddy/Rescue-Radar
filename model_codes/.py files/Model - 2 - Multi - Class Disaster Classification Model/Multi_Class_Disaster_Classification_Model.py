# Importing Packages
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image, ImageFile
import tensorflow as tf
import json
import glob
import random
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import gradio as gr
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pathlib
dataset_path = "/kaggle/working/train"
train_data_dir = pathlib.Path(dataset_path + "/images")
test_data_dir = pathlib.Path("/kaggle/working/test/images")

train_files = glob.glob(r"" + dataset_path + "/images/*.png")
train_files = list(filter(lambda x: "post" in x, train_files))
train_files = random.sample(train_files, 1500)
train_datasize = len(train_files)
print("Training data:", train_datasize)

test_files = glob.glob(r"" + "/kaggle/working/test/images/*.png")
test_files = list(filter(lambda x: "post" in x, test_files))
test_files = random.sample(test_files, 500)
test_datasize = len(test_files)
print("Test data:", test_datasize)

images = list(train_data_dir.glob('*'))
random_image = random.choice(images)
im = PIL.Image.open(str(random_image))

width, height = im.size
print(width)
print(height)
im.resize((300, 300)).show()

img_height = 1024
img_width = 1024
class_names = np.array(sorted(['volcano', 'flooding', 'earthquake', 'fire', 'wind', 'tsunami']))
print(class_names)

def get_label(file_path, type):
    parts = file_path.split('/')
    path = dataset_path + '/labels/'
    if type == "test":
        path = '/kaggle/working/test/labels/'
    f = open(path + parts[-1].split('.')[0] + '.json')
    data = json.load(f)
    disaster_type = data['metadata']['disaster_type']
    f.close()

    label = disaster_type == class_names
    one_hot = np.zeros(len(class_names), dtype=np.uint8)
    one_hot[label] = 1

    return one_hot

def get_label_from_one_hot(array):
    return class_names[np.where(array == 1)]

train_X = np.zeros((train_datasize, img_height, img_width, 3), dtype=np.uint8)
train_Y = np.zeros((train_datasize, len(class_names)), dtype=np.uint8)

for i in range(len(train_files)):
    img = PIL.Image.open(train_files[i])
    train_X[i] = np.array(img)
    train_Y[i] = get_label(train_files[i], "train")
print("Train")
print(train_X.shape)
print(train_Y.shape)

test_X = np.zeros((test_datasize, img_height, img_width, 3), dtype=np.uint8)
test_Y = np.zeros((test_datasize, len(class_names)), dtype=np.uint8)

for i in range(len(test_files)):
    img = PIL.Image.open(test_files[i])
    test_X[i] = np.array(img)
    test_Y[i] = get_label(test_files[i], "test")
print("Test")
print(test_X.shape)
print(test_Y.shape)

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    choice = random.randint(0, train_datasize - 1)
    plt.title(get_label_from_one_hot(train_Y[choice]))
    plt.imshow(train_X[choice])

plt.tight_layout()
plt.show()

# Load the DenseNet121 model, exclude the top layers (pre-trained on ImageNet)
base_model = DenseNet121(input_shape=(img_height, img_width, 3),
                         include_top=False,
                         weights='imagenet')

base_model.trainable = False  # Freeze the base model

model = models.Sequential([
    layers.Input(shape=(1024, 1024, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Define the maximum number of epochs
max_epochs = 50
batch_size = 16

# Create an EarlyStopping callback with patience
early_stopping = EarlyStopping(
    monitor='val_loss',# Metric to monitor
    patience = 5,
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric
)

# Train the model with EarlyStopping
history = model.fit(
    train_X, train_Y,
    validation_data=(test_X, test_Y),
    epochs=max_epochs,
    batch_size=batch_size,
    callbacks=[early_stopping]  # Pass the callback
)

# Model Evaluation and Metrics Calculation
#y_pred = model.predict(test_X)
#y_pred_classes = np.argmax(y_pred, axis=1)
#y_true_classes = np.argmax(test_Y, axis=1)

# Calculate accuracy
#accuracy = accuracy_score(y_true_classes, y_pred_classes)
#print(f"Accuracy: {accuracy:.4f}")

# Evaluate model on test data to get loss and accuracy
print("\nEvaluating model on test data:\n")
loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)

# Print loss and accuracy
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Generate classification report
class_report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0)
print("Classification Report:")
print(class_report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(max_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save as a Keras model (Native Keras format)
model.save('Multi_Class_Disaster_Classification_Model.keras')

# Save as a .h5 file (HDF5 format), explicitly including the optimizer
model.save('Multi_Class_Disaster_Classification_Model.h5', include_optimizer=True)

# Export as TensorFlow SavedModel format (directory format)
model.export('Multi_Class_Disaster_Classification_Model')

# Gradio Interface for Inference
def disaster_classification(img):
    img_resized = np.array(Image.fromarray(img).resize((img_height, img_width)))
    image = np.zeros((1, img_height, img_width, 3), dtype=np.uint8)
    image[0] = img_resized
    prediction = model.predict(image).tolist()[0]
    return {class_names[i]: prediction[i] for i in range(len(class_names))}

iface = gr.Interface(
    fn=disaster_classification,
    inputs=gr.Image(image_mode='RGB', type='numpy'),
    outputs=gr.Label()
)

iface.launch(share=True, show_error=True, debug=True)