"""A demo for object detection or image classification using CORAL TPU.

This example is intended to run later in a raspberry PI, but for now, is running on a
Linux machine

The only pending thing to make it run on the raspberry, since capturing frames require
a different method through the picamera python library
See:
https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python

For running in a Linux PC, follow the standard installation of the CORAL TPU USB, plus
installing Python-OpenCV

Examples (Running under python-tflite-source/edgetpu directory):
  - Object recognition:
    python3 demo/my_TPU_image_recognition.py \
    --model=test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
    --label=test_data/coco_labels.txt --mode=OBJECT_DETECTION \
    --camera=0

"""
import pantilthat
import argparse
import platform
import subprocess
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw
import numpy as np
import time
from collections import deque, Counter

#For webcam capture and drawing boxes
import cv2

#picamera
from picamera import PiCamera
import io

# Parameters for visualizing the labels and boxes
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.7
LABEL_BOX_PADDING = 5
LABEL_BOX_OFFSET_TOP = int(20 * FONT_SIZE) + LABEL_BOX_PADDING
LINE_WEIGHT = 1
IMAGE_SIZE = (640,480)
IMAGE_CENTER = (int(IMAGE_SIZE[0]/2), int(IMAGE_SIZE[1]/2))
MINANGLESTEP = 1 #Degrees
IMGTHRESHOLD = 20 #pixels to avoid flickering

# Function to read labels from text files.
def read_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='Path of the detection model.', required=True)
    parser.add_argument(
        '--label', help='Path of the labels file.')
#    parser.add_argument(
#        '--mode', help='Mode for de detection: OBJECT_DETECTION or IMAGE_CLASSIFICATION',
#        required=True)
    parser.add_argument(
        '--camera', help='Camera source (if multiple available)', type=int, required=False)

    args = parser.parse_args()

    # Initialize the pan-tilt thing
    pantilthat.pan(0)
    pantilthat.tilt(0)

    # Initialize engine.

    engine = DetectionEngine(args.model)

    labels = read_label_file(args.label) if args.label else None
    label = None
    camera = args.camera if args.camera else 0

    # Initialize the camera
    #cam = cv2.VideoCapture(camera)
    camera = PiCamera()
    time.sleep(2)
    camera.resolution = IMAGE_SIZE
    camera.vflip = True
    # Create the in-memory stream
    stream = io.BytesIO()

    # Initialize the timer for fps
    start_time = time.time()
    frame_times = deque(maxlen=40)

    screencenter = (int(IMAGE_SIZE[0]/2), int(IMAGE_SIZE[1]/2))

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

        if True:
            ans = engine.DetectWithImage(pil_im, threshold=0.05, keep_aspect_ratio=True,
                                         relative_coord=False, top_k=10)
            if ans:
                for obj in ans:
                    if obj.score > 0.4:
                        #if labels:
                        #    label = labels[obj.label_id] + " - {0:.2f}".format(obj.score)
                        #draw_rectangles(obj.bounding_box, cv2_im, label=label)
                        resultado = show_box_center_and_size(obj.bounding_box)
                        stepx, stepy = center_camera(resultado, screencenter)
            else:
                #draw_text(cv2_im, 'No object detected!')
                pass

        frame_times.append(time.time())
        fps = len(frame_times)/float(frame_times[-1] - frame_times[0] + 0.001)
        #draw_text(cv2_im, "{:.1f} / {:.2f}ms".format(fps, lastInferenceTime))
        #print("FPS / Inference time: " + "{:.1f} / {:.2f}ms".format(fps, lastInferenceTime))

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
    threshold = IMGTHRESHOLD  # 10 by default

    dX = screencenter[0] - objxy[0]
    dY = screencenter[1] - objxy[1]

    stepx = int(abs(dX)/10 - IMGTHRESHOLD/10) 
    stepy = int(abs(dY)/10 - IMGTHRESHOLD/10)


    currentPan = pantilthat.get_pan()
    currentTilt = pantilthat.get_tilt()

    print(f"({objxy}) status: pan:{currentPan}, tilt:{currentTilt}; (dX:{dX}, dy:{dY}, step:{stepx})")

    if dX < 0 - IMGTHRESHOLD and abs(currentPan) < max_angle + stepx:
       pantilthat.pan(currentPan + stepx)
    elif dX > 0 + IMGTHRESHOLD and abs(currentPan) < max_angle + stepx:
       pantilthat.pan(currentPan - stepx)

    if dY < 0 - IMGTHRESHOLD and abs(currentPan) < max_angle + stepy:
       pantilthat.tilt(currentTilt + stepy)
    elif dY > 0 + IMGTHRESHOLD and abs(currentPan) < max_angle + stepy: 
       pantilthat.tilt(currentTilt - stepy)

    return (stepx, stepy)

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

    #print(IMAGE_CENTER[0] - centerX, IMAGE_CENTER[1] - centerY, w, h)
    return (centerX, centerY)

def draw_rectangles(rectangles, image_np, label=None):
    p1 = (int(rectangles[0][0]), int(rectangles[0][1]))
    p2 = (int(rectangles[1][0]), int(rectangles[1][1]))
    cv2.rectangle(image_np, p1, p2, color=(255, 0, 0), thickness=LINE_WEIGHT)
    if label:
        cv2.rectangle(image_np, (p1[0], p1[1]-LABEL_BOX_OFFSET_TOP), (p2[0], p1[1] + LABEL_BOX_PADDING),
                      color=(255, 0, 0),
                      thickness=-1)
        cv2.putText(image_np, label, p1, FONT, FONT_SIZE, (255, 255, 255), 1, cv2.LINE_AA)
    #imgname = str(time.time())
    #cv2.imwrite('/home/pi/development/Coral-TPU/imgs/' + imgname + '.jpg', image_np)

def draw_text(image_np, label, pos=0):
    p1 = (0, pos*30+20)
    #cv2.rectangle(image_np, (p1[0], p1[1]-20), (800, p1[1]+10), color=(0, 255, 0), thickness=-1)
    cv2.putText(image_np, label, p1, FONT, FONT_SIZE, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == '__main__':
    main()
