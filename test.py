import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np
import os

# Load the pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)

num_classes = len(os.listdir('path/to/your/dataset'))
# Add custom layers for your task
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Create the final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to make predictions on a single image
def predict_image(file_path):
    img = image.load_img(file_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode predictions (if using a model with ImageNet classes)
    decoded_predictions = decode_predictions(predictions)

    # Display the results
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{label}: {score}")

    # Count the occurrences of each class
    class_counts = {label: decoded_predictions[0][i][2] for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0])}

    # Print the counts
    for label, count in class_counts.items():
        print(f"{label}: {count}")

# Provide the path to your image file
image_file_path = 'apple.jpeg'

# Call the predict_image function
predict_image(image_file_path)
