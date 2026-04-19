import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.face_detection import detect_face
from utils.feature_extraction import extract_features

# Load trained model
model = load_model("model/deepfake_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)

    for face, (x,y,w,h) in faces:
        try:
            features = extract_features(face)
            features = np.expand_dims(features, axis=0)

            prediction = model.predict(features)[0][0]

            label = "Fake" if prediction > 0.5 else "Real"
            color = (0,0,255) if label=="Fake" else (0,255,0)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{label} ({prediction:.2f})",
                        (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        except:
            continue

    cv2.imshow("Real-Time Deepfake Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()