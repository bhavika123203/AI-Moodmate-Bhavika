import tensorflow as tf
import cv2

def load_face_model():
    """Loads the entire .keras model from the file."""
    model = tf.keras.models.load_model('models/mobilenetmodel.keras')
    return model

def load_face_detector():
    """Loads the Haarcascade classifier for face detection."""
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_classifier.empty():
        print("Haarcascade Classifier failed to load.")
    return face_classifier

