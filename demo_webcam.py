from tensorflow import lite
import numpy as np
import cv2

MODEL_PATH = "models/model_full_integer_quant.tflite"
INPUT_SHAPE = (256, 256)
KEY_POINTS = 16
confidence = 0

# Load model
model = lite.Interpreter(model_path=MODEL_PATH)
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, cap_frame = cap.read()
    cap_frame = cv2.resize(cap_frame, INPUT_SHAPE)
    batch = np.expand_dims(cap_frame, axis=0).astype(np.int8)

    # Perform inference
    model.set_tensor(input_details[0]['index'], batch)
    model.invoke()
    batch_outputs = model.get_tensor(output_details[-1]['index'])

    # Extract coordinates
    coordinates = list()
    for idx in range(KEY_POINTS):
        frame = batch_outputs[0][:, :, idx]
        if np.amax(frame) < confidence: continue
        row, col = np.unravel_index(np.argmax(frame, axis=None), frame.shape)
        coordinates.append((row, col))
        cv2.circle(cap_frame, (col, row), 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("EfficientPose", cap_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()