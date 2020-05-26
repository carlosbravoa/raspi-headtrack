#!/usr/bin/env python

"""A demo for object detection or image classification using CORAL TPU.

This has been written "quick and dirty", but it works and is very fun to play with.

Run it with:
python3 head_tracking_pantilt.py \\
    --model models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
"""
import io
import argparse
import time
from PIL import Image
from edgetpu.detection.engine import DetectionEngine

import pantilthat
from picamera import PiCamera

# Parameters for visualizing the labels and boxes
IMAGE_SIZE = (640, 480)
IMAGE_CENTER = (int(IMAGE_SIZE[0]/2), int(IMAGE_SIZE[1]/2))
MINANGLESTEP = 1 #Degrees
IMGTHRESHOLD = 20 #pixels to stop moving once the image has been more or less centered

def main():
    '''Main function for running head tracking.
    It will initialize picam, CoralTPU with a given model (e.g. face recorgnition)
    Then, it will capture an image, feed it into CoralTPU, get coordinates and move servos
    (pan, tilt) towards the center of the box
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='Path of the detection model.', required=True)

    args = parser.parse_args()

    # Initialize the pan-tilt thing
    pantilthat.pan(0)
    pantilthat.tilt(0)

    # Initialize engine.

    engine = DetectionEngine(args.model)

    # Initialize the camera
    #cam = cv2.VideoCapture(camera)
    camera = PiCamera()
    time.sleep(2)
    camera.resolution = IMAGE_SIZE
    camera.vflip = True
    # Create the in-memory stream
    stream = io.BytesIO()

    print("Capture started")
    while True:
        #ret, cv2_im = cam.read()
        stream = io.BytesIO() #wipe the contents
        camera.capture(stream, format='jpeg', use_video_port=True)
        stream.seek(0)
        # we need to flip it here!
        pil_im = Image.open(stream)
        #cv2_im = np.array(pil_im)
        #cv2_im = cv2.flip(cv2_im, 1) #Flip horizontally
        #cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

        ans = engine.DetectWithImage(pil_im, threshold=0.05, keep_aspect_ratio=True,
                                     relative_coord=False, top_k=10)
        if ans:
            for obj in ans:
                if obj.score > 0.4:
                    resultado = show_box_center_and_size(obj.bounding_box)
                    center_camera(resultado, IMAGE_CENTER)
        else:
            pass

def center_camera(objxy, screencenter):
    '''
    from an (X,Y), it tries to pan/tilt to minimize the distance with the center of the "view"

    NOTE: From the viewer's pov: If face is upper left:
                                      -> dX < 0 & dY >0
                                      -> pan++ aims left, tilt-- aims up

    Args:
        objxy: Tuple (x,y) of the detected object
        screencenter: Tuple(x,y) with the center of the screen
    '''

    max_angle = 80  # To stay safe and not exeed the max angle of the servo
    stepx = MINANGLESTEP  # 1 by default #+ abs(int(objxy[0]/100))
    stepy = MINANGLESTEP

    dX = screencenter[0] - objxy[0]
    dY = screencenter[1] - objxy[1]

    stepx = int(abs(dX)/10 - IMGTHRESHOLD/10)  # Empirical value. 10% of the distance, good enough
    stepy = int(abs(dY)/10 - IMGTHRESHOLD/10)

    currentPan = pantilthat.get_pan()
    currentTilt = pantilthat.get_tilt()

    if dX < 0 - IMGTHRESHOLD and abs(currentPan) < max_angle:
        pan_direction = 1

    elif dX > 0 + IMGTHRESHOLD and abs(currentPan) < max_angle:
        pan_direction = -1

    if dY < 0 - IMGTHRESHOLD and abs(currentPan) < max_angle:
        tilt_direction = 1

    elif dY > 0 + IMGTHRESHOLD and abs(currentPan) < max_angle:
        tilt_direction = -1

    newPan = currentPan + pan_direction * stepx  # Add or substract stepx to pan
    newPan = newPan % pan_direction * max_angle  # To avoid having a value higher than max_angle

    newTilt = currentTilt + tilt_direction * stepy  # Add or substract stepy to tilt
    newTilt = newTilt % tilt_direction * max_angle  # To avoid having a value higher than max_angle

    pantilthat.pan(newPan)
    pantilthat.tilt(newTilt)

def show_box_center_and_size(rectangle):
    '''
    Returns the center of the bounding box received
    Args:
        rectangle: list with two lists inside: [[X0Y0],[X1Y1]]
    Returns:
        a tuple with CenterX and CenterY
    '''

    X0 = rectangle[0][0]
    X1 = rectangle[1][0]
    Y0 = rectangle[0][1]
    Y1 = rectangle[1][1]

    width = X1 - X0
    centerX = int(X0 + (width / 2.0))

    h = Y1 - Y0
    centerY = int(Y0 + (h / 2.0))

    return (centerX, centerY)

if __name__ == '__main__':
    main()
