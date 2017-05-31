from flask import Flask, request, render_template, jsonify, make_response
import traceback
import logging
import tensorflow as tf
import cv2
import numpy as np
import base64
from face_convert import FaceConvert

convert = FaceConvert()


app = Flask(__name__)
logger = app.logger
logger.setLevel(logging.INFO)
hdr = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
hdr.setFormatter(formatter)
logger.addHandler(hdr)


def detect(img_bytes):
    img = cv2.imdecode(np.asarray(bytearray(img_bytes), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    boxes = convert.detect_face(img)
    for b in boxes.astype(np.uint32):
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img


@app.route("/detect", methods=['GET', 'POST'])
def face_detect():
    if request.method == 'POST':
        f = request.files['face']
        img_bytes = f.stream.read()
        # img_np = cv2.imdecode(np.asarray(bytearray(img_bytes), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        output_img = detect(img_bytes)
        retval, bf = cv2.imencode('.jpg', output_img)
        img_as_text = base64.b64encode(bf)
        response = img_as_text
    else:
        response = render_template("detect.html")

    return response


@app.route("/swap", methods=['GET', 'POST'])
def swap_face():
    if request.method == 'GET':
        response = render_template("swap.html")
    else:
        f1 = request.files['face1']
        f2 = request.files['face2']
        img_np1 = cv2.imdecode(np.asarray(bytearray(f1.stream.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img_np2 = cv2.imdecode(np.asarray(bytearray(f2.stream.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        output_img = convert.swap_face(img_np1, img_np2)
        retval, bf = cv2.imencode('.jpg', output_img)
        img_as_text = base64.b64encode(bf)
        response = img_as_text
    return response


@app.errorhandler(500)
def handle_error(e):
    logger.exception('error 500: %s', e)
    return "出错了！"


@app.route("/")
def hello():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=False)
