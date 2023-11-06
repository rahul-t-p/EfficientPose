from tensorflow import lite
import numpy as np
import cv2
import argparse

MODEL_DIR = "models"
MODEL_NAME = "model_full_integer_quant.tflite"
INPUT_SHAPE_DICT = {"RT_LITE": (224, 224), "I_LITE": (256, 256), "II_LITE": (368, 368)}
CONFIDENCE_DICT = {"RT_LITE": -80, "I_LITE": 0, "II_LITE": -50}
KEY_POINTS = 16

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Run inference
def run_inference(model_version: str):
    # Load model
    model = lite.Interpreter(model_path="{}/{}/{}".format(MODEL_DIR, model_version, MODEL_NAME))
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Get width and height of webcam input
    cap_height, cap_width = cap.read()[1].shape[:2]

    while True:
        # Capture a frame from the webcam
        ret, cap_frame = cap.read()
        cap_frame = cv2.resize(cap_frame, INPUT_SHAPE_DICT[model_version])
        batch = np.expand_dims(cap_frame, axis=0).astype(np.int8)

        # Perform inference
        model.set_tensor(input_details[0]['index'], batch)
        model.invoke()
        batch_outputs = model.get_tensor(output_details[-1]['index'])

        # Extract coordinates
        for idx in range(KEY_POINTS):
            frame = batch_outputs[0][:, :, idx]
            if np.amax(frame) < CONFIDENCE_DICT[model_version]: continue
            x, y = np.unravel_index(np.argmax(frame, axis=None), frame.shape)
            cv2.circle(cap_frame, (y, x), 3, (0, 0, 255), -1)

        # Display the frame
        cap_frame = cv2.resize(cap_frame, (cap_width, cap_height))
        cv2.imshow("EfficientPose", cap_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, default="RT_LITE", help="The version of the EfficientPose to use -> RT_LITE (default) / I_LITE / II_LITE")
    args = parser.parse_args()
    if args.model_version not in INPUT_SHAPE_DICT.keys():
        parser.print_help()
        exit(-1)

    print("\n{}\nRunning EfficinetPose{}\n{}\n".format("*"*50, args.model_version, "*"*50))
    print("Press \'q\' to exit...\n")

    run_inference(args.model_version)

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()
