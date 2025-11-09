import cv2
import sys
import numpy as np
from tensorflow.keras import models

MODEL_PATH = "tf-cnn-model.h5"

def predict_digit(image_path):
    # Load model (ignore training config)
    model = models.load_model(MODEL_PATH, compile=False)
    print("[INFO] Loaded model from disk.")

    # Read and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("[ERROR]: Invalid image path.")
        return None

    image_resized = cv2.resize(image, (28, 28))
    image_input = image_resized.reshape(1, 28, 28, 1) / 255.0  # normalize

    # Display image
    cv2.imshow('Digit', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Predict
    pred = np.argmax(model.predict(image_input), axis=-1)
    return pred[0]

def main(image_path):
    predicted_digit = predict_digit(image_path)
    if predicted_digit is not None:
        print('Predicted Digit: {}'.format(predicted_digit))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_model.py <image_path>")
    else:
        main(sys.argv[1])
