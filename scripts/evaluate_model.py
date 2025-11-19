import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Use raw strings or double backslashes for Windows paths
validation_dir = r'C:\Users\Vishnu R Shetty\OneDrive\ドキュメント\HTML\CSS\PROJECTS\foot_ulcer_detection\foot_ulcer_detection\dataset\validation'

# Provide the full path to the actual .h5 model file (not the folder)
model_path = r'C:\Users\Vishnu R Shetty\OneDrive\ドキュメント\HTML\CSS\PROJECTS\foot_ulcer_detection\foot_ulcer_detection\models'
model = tf.keras.models.load_model(model_path)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy:.2f}')
print(f'Validation loss: {loss:.2f}')
