from itertools import chain

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def create_labels(detection_result):
    # 21 points * 3 corr * 2 hands
    points = [0] * 21 * 3 * 2
    scores = {"Left": 0, "Right": 0}

    if not detection_result:
        return points, scores

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    for handedness, landmarks in zip(handedness_list, hand_landmarks_list):
        index, score, name, _ = handedness[0].__dict__.values()

        landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks]
        landmarks = list(chain.from_iterable(landmarks))

        if name == "Left":
            points[:63] = landmarks
            scores["Left"] = score

        if name == "Right":
            points[63:] = landmarks
            scores["Right"] = score

    return points, scores


class MPCallback:
    RESULT = None
    VIDEO = None

    def callback(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ):
        self.RESULT = result
        self.VIDEO = self.draw_landmarks(output_image.numpy_view(), result)

    def draw_landmarks(self, image, detection_result):
        annotated_image = np.copy(image)

        if not detection_result:
            return annotated_image

        hand_landmarks_list = detection_result.hand_landmarks

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

        return annotated_image


callback = MPCallback()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=callback.callback,
)

with HandLandmarker.create_from_options(options) as landmarker:
    # Load trained SVM model and scaler
    svm_model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Drop to square
        h, w = frame.shape[:2]
        size = min(h, w)
        x, y = (w - size) // 2, (h - size) // 2

        image = np.array(frame[y : y + size, x : x + size])

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        landmarker.detect_async(mp_image, frame_timestamp_ms)

        points, scores = create_labels(callback.RESULT)

        if sum(points) != 0:
            # Standardize using the same scaler
            features_scaled = scaler.transform([points])

            # Make prediction
            prediction = svm_model.predict(features_scaled)[0]
        else:
            prediction = "..."

        # Display prediction on the frame
        label_text = f"Handsign: {prediction}"

        print(
            f"Left: {scores['Left']:.2f} Right: {scores['Right']:.2f} {label_text}  ",
            end="\r",
        )

        cv2.putText(
            callback.VIDEO, label_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3
        )

        if callback.VIDEO is None:
            continue

        # Show the frame
        cv2.imshow("Videoframe", callback.VIDEO)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
