import cv2
import tensorflow as tf
import numpy as np
from tkinter import Tk
from tkinter import filedialog

# Load pre-trained model for fruit/vegetable classification
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # adjust size based on the model input size
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

def classify_fruit_vegetable(img):
    processed_image = preprocess_image(img)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    return decoded_predictions[0][0][1]

def estimate_weight(fruit_vegetable):
    # This is a fictional function, and you may need a more sophisticated solution
    # based on a weight estimation model or database.
    weights_database = {
        'apple': 150,  # in grams
        'banana': 120,
        'orange': 200,
        # Add more fruits/vegetables and their weights
    }
    return weights_database.get(fruit_vegetable, "Unknown")

def main():
    print("Choose an option:")
    print("1. Take a picture")
    print("2. Use an image from the gallery")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        # Open the camera
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1)
            if key == 27:  # Press 'Esc' to exit
                break
            elif key == 32:  # Press 'Space' to capture image
                cv2.imwrite("captured_image.jpg", frame)
                break

        cap.release()
        cv2.destroyAllWindows()

        image_path = "captured_image.jpg"

    elif choice == '2':
        # Use file dialog to select an image from the gallery
        Tk().withdraw()  # Hide the main window
        image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    else:
        print("Invalid choice. Please enter either '1' or '2'.")
        return

    try:
        img = cv2.imread(image_path)
        fruit_vegetable = classify_fruit_vegetable(img)
        print(f"Detected fruit/vegetable: {fruit_vegetable}")

        estimated_weight = estimate_weight(fruit_vegetable)
        if estimated_weight != "Unknown":
            print(f"Estimated weight: {estimated_weight} grams")
        else:
            print("Weight estimation not available for this item.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
