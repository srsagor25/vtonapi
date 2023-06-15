from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import cvzone
from cvzone.PoseModule import PoseDetector
import os

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
detector = PoseDetector()

shirt_folder_path = "Resources/Shirts"
list_shirts = os.listdir(shirt_folder_path)
fixed_ratio = 250 / 190  # widthOfShirt/widthOfPoint11to12
shirt_ratio_height_width = 581 / 440
image_number = 1

default_shirt_width = 200
current_image = None


@app.route('/', methods=['GET', 'POST'])
def index():
    global current_image
    if request.method == 'POST':
        image = request.files['image']
        image.save('received_image.jpg')
        current_image = cv2.imread('received_image.jpg')
        return 'Image received successfully'
    else:
        return render_template('index2.html')


def process_frame():
    global current_image
    image_number = 1
    hand_raise_duration = 5
    left_hand_raised_time = 0
    right_hand_raised_time = 0
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findPose(img)
        lmList, bbox_info = detector.findPosition(img, bboxWithHands=False, draw=False)

        if current_image is not None:
            # Process the current image received from Flutter
            img_shirt = current_image.copy()

            if lmList:
                lm11 = lmList[11][1:4]
                lm12 = lmList[12][1:4]
                lm23 = lmList[23][1:4]
                lm24 = lmList[24][1:4]
                # Calculate the desired width of the shirt based on the distance between landmarks
                width_of_shirt = int((lm11[0] - lm12[0]) * fixed_ratio)
                length_of_shirt = int(abs(lm11[1] - lm23[1]) * fixed_ratio)
                # Check if the width is valid and the person is facing the front
                if width_of_shirt > 0 and lmList[11][2] < lmList[23][2]:
                    img_shirt = cv2.resize(img_shirt, (width_of_shirt, length_of_shirt))
                    current_scale = (lm11[0] - lm12[0]) / 190
                    offset = int(44 * current_scale), int(48 * current_scale)

                    try:
                        img = cvzone.overlayPNG(img, img_shirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
                    except:
                        pass

            current_image = None

        # Convert the frame to JPEG and yield it
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

    # Run the Flask application on a specific IP address
    app.run(host='192.168.0.147', port=5000, debug=True)