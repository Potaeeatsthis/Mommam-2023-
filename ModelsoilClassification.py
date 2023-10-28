import cv2
import numpy as np
import tensorflow as tf
import time
import os
import shutil
from pymongo import mongo_client

interpreter = tf.lite.Interpreter(
    model_path="model_saved/trained_pic_classify_model.tflite")
interpreter.allocate_tensors()

class_label = ['Black Soil', 'Cinder Soil',
               'Laterite Soil', 'Peat Soil', 'Yellow Soil']

left_x, left_y, region_width, region_height = 200, 80, 500, 500
right_x, right_y = 700, 80

left_screenshot_count = 0
right_screenshot_count = 0

output_folder = "capture_pic"
os.makedirs(output_folder, exist_ok=True)

def preprocess_frame(frame):
    if frame is None or frame.size == 0:
        return None

    resized_frame = cv2.resize(frame, (220, 220))
    normalized_frame = resized_frame / 255.0
    return normalized_frame


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()

if not ret:
    print("Error: Could not read the first frame.")
    exit()

frame_height = int(cap.get(3))
frame_width = int(cap.get(4))
half_point = frame_height // 2
half_width = frame_width // 2

tracker_right = cv2.TrackerCSRT_create()
tracker_left = cv2.TrackerCSRT_create()

right_init_rect = (100, half_width, 100, 100)
left_init_rect = (half_point + 100, half_width, 100, 100)

tracker_right.init(frame, right_init_rect)
tracker_left.init(frame, left_init_rect)

screenshot_interval = 5

max_duration = 7

last_screenshot_time = time.time()
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    ret_right, right_rect = tracker_right.update(frame)
    ret_left, left_rect = tracker_left.update(frame)

    left_frame = frame[left_y:left_y +
                       region_height, left_x:left_x + region_width]
    right_frame = frame[right_y:right_y +
                        region_height, right_x:right_x + region_width]

    if ret_right:
        (x, y, w, h) = [int(i) for i in right_rect]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        roi = frame[y:y + h, x:x + w]
        processed_roi = preprocess_frame(roi)

        if processed_roi is not None:
            input_details = interpreter.get_input_details()
            input_data = np.expand_dims(
                processed_roi, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = class_label[np.argmax(output_data)]

            cv2.putText(frame, f'Soil: {predicted_class}', (x, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)

    if ret_left:
        (x, y, w, h) = [int(i) for i in left_rect]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 255, 255), 2)
        roi = frame[y:y + h, x:x + w]
        processed_roi = preprocess_frame(roi)

        if processed_roi is not None:
            input_details = interpreter.get_input_details()

            input_data = np.expand_dims(
                processed_roi, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = class_label[np.argmax(output_data)]

            cv2.putText(frame, f'Soil: {predicted_class}', (x, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)

    current_time = time.time()

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if time.time() - last_screenshot_time > screenshot_interval:
        left_screenshot_count += 1
        right_screenshot_count += 1

        left_screenshot_count
        right_screenshot_count

        screenshot_filename_left = os.path.join(
            output_folder, f"left_screenshot_{left_screenshot_count}.png")
        screenshot_filename_right = os.path.join(
            output_folder, f"right_screenshot_{right_screenshot_count}.png")

        cv2.imwrite(screenshot_filename_left, left_frame)
        cv2.imwrite(screenshot_filename_right, right_frame)

        print(f"Saved left screenshot as {screenshot_filename_left}")
        print(f"Saved right screenshot as {screenshot_filename_right}")

        last_screenshot_time = time.time()

    if time.time() - start_time >= max_duration:
        break

cap.release()
cv2.destroyAllWindows()

# files_to_copy = [
#     "capture_pic/left_screenshot_1.png",
#     "capture_pic/right_screenshot_1.png",
# ]


# flash_drive_name = "AI"

# flash_drive_path = os.path.join("/Volumes", flash_drive_name)


# if not os.path.exists(flash_drive_path):
#     print(f"Flash drive '{flash_drive_name}' is not mounted.")
# else:
#     try:
#         for source_file in files_to_copy:
#             if os.path.exists(source_file):
#                 filename = os.path.basename(source_file)
#                 shutil.copy(source_file, os.path.join(
#                     flash_drive_path, filename))
#                 print(f"Copied '{filename}' to flash drive.")
#             else:
#                 print(f"Source file '{source_file}' does not exist.")
#         print("Done copying files to flash drive.")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

client = MemoryError("mongodb+srv://<ninninrapat>:<Ninnin081020>@<cluster-url>/<database>?retryWrites=true&w=majority")