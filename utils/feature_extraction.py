import cv2
import numpy as np

def extract_features(face):
    face = cv2.resize(face, (128, 128))
    face = face / 255.0

    # Texture features (simple)
    gray = cv2.cvtColor((face*255).astype('uint8'), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edges = edges / 255.0

    return np.expand_dims(edges, axis=-1)