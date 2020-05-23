# raspi-headtrack
Using a raspberry pi, Coral TPU, pan-tilt hat, I wrote a simple code for head tracking.

![Raspberry PI with pan-tilt hat and CoralTPU](pan-tilt.png)

Requirements: 
- Coral TPU (and modules installed)
- Pan-tilt hat (and modules installed)

Run it with:
python3 head_tracking_pantilt.py --model models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

