# predict.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from build_train_model import build_advanced_model  # Ensure this is the correct import
from tensorflow.keras.models import load_model

def select_random_image(folder_path):
    """
    Select a random image from the specified folder.
    """
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    random_image = random.choice(images)
    return os.path.join(folder_path, random_image)

def load_and_prepare_image(image_path, img_size=48):
    """
    Load and preprocess an individual image for model prediction.
    """
    img = image.load_img(image_path, target_size=(img_size, img_size), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_prediction(model, prepared_image, class_labels):
    """
    Make a prediction using the model and return the corresponding label.
    """
    prediction = model.predict(prepared_image)
    predicted_class = np.argmax(prediction, axis=1)
    return class_labels[predicted_class[0]]

def main():
    parent_directory = os.path.dirname(os.getcwd())
    emotion = 'happy'  # Example
    folder_path = os.path.join(parent_directory, 'data', 'train', emotion)
    image_path = select_random_image(folder_path)

    class_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    prepared_image = load_and_prepare_image(image_path)
    model_path = os.path.join(parent_directory, 'models', 'final_model_w_weights.h5')
    model = load_model(model_path)

    predicted_label = make_prediction(model, prepared_image, class_labels)

    # Display the image and prediction
    img = image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted Emotion: {predicted_label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
