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
import sys
from PIL import Image
from edgetpu.detection.engine import DetectionEngine

import pantilthat
from picamera import PiCamera

# Parameters for visualizing the labels and boxes
IMAGE_SIZE = (640, 480)
IMAGE_CENTER = (int(IMAGE_SIZE[0]/2), int(IMAGE_SIZE[1]/2))
MINANGLESTEP = 1 #Degrees
IMGTHRESHOLD = 20 #pixels to stop moving once the image has been more or less centered
DEBUG = False # Change this to see more information on console about the detection

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
    reset_pan_tilt()

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

    debug = DEBUG
    image_center = IMAGE_CENTER

    print("Capture started")
    while True:
        try:
            #ret, cv2_im = cam.read()
            stream = io.BytesIO() #wipe the contents
            camera.capture(stream, format='jpeg', use_video_port=True)
            stream.seek(0)
            # we need to flip it here!
            pil_im = Image.open(stream)
            #cv2_im = np.array(pil_im)
            #cv2_im = cv2.flip(cv2_im, 1) #Flip horizontally
            #cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            object_on_sight = False
            timer = None

            ans = engine.DetectWithImage(pil_im, threshold=0.3, keep_aspect_ratio=True,
                                        relative_coord=False, top_k=1)
            if ans:
                #for obj in ans:  # For multiple objects on screen
                obj = ans[0]  # Follow only one object, the first detected object

                object_on_sight = True
                timer = None
                result = show_box_center_and_size(obj.bounding_box)
                center_camera(result, image_center, debug)

            else:
                #print("No objects detected with the given threshold")
                object_on_sight = False

            if not object_on_sight and not timer:  # if there was an object before, and is no longer on screen
                timer = time.time()  # We start a timer for reseting the angles later
            
            if timer and not object_on_sight:
                elapsed_time = time.time() - timer
                # If 5 seconds have passed without activity. More than 8, do nothing
                if elapsed_time > 5 and elapsed_time < 8:
                    reset_pan_tilt()
                    timer = None  # We stop the timer

        except KeyboardInterrupt:
            print("Closing program")
            reset_pan_tilt()
            sys.exit()

        except:
            reset_pan_tilt()
            raise

def reset_pan_tilt():
    pantilthat.pan(0)
    pantilthat.tilt(0)
    time.sleep(1)

def center_camera(objxy, screencenter, debug=False):
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

    # Computing the next angle, starting on the current
    newPan = currentPan
    newTilt = currentTilt

    if dX < 0 - IMGTHRESHOLD: # and abs(currentPan) + stepx < max_angle:
        newPan = currentPan + stepx
        newPan = newPan if newPan < max_angle else max_angle
        
    elif dX > 0 + IMGTHRESHOLD: # and abs(currentPan) + stepx < max_angle:
        newPan = currentPan - stepx
        newPan = newPan if newPan > -max_angle else -max_angle

    if dY < 0 - IMGTHRESHOLD: # and abs(currentTilt) + stepy < max_angle:
        newTilt = currentTilt + stepy
        newTilt = newTilt if newTilt < max_angle else max_angle

    elif dY > 0 + IMGTHRESHOLD: # and abs(currentTilt) + stepy < max_angle:
        newTilt = currentTilt - stepy
        newTilt = newTilt if newTilt > -max_angle else -max_angle

    if debug:
        print(f"({objxy}) status: pan:{currentPan}, tilt:{currentTilt}; (dX:{dX}, dy:{dY}, step:{stepx},{stepy})({newPan},{newTilt})")

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
