
import os
import dlib
import cv2
import tensorflow as tf
import numpy as np
from facenet.align.detect_face import detect_face ,create_mtcnn
import pylab as plt


PREDICTOR_PATH = "pretrain_model/dlib/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
SCALE_FACTOR = 2
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR_FRAC = 0.9


FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)


class NoFaces(Exception):
    pass


def transformation_from_points(points1, points2):
    points1 = np.matrix(points1)
    points2 = np.matrix(points2)
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    mask = np.zeros(im.shape[:2], dtype=np.float64)
    draw_convex_hull(mask,landmarks[LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS],color=1)
    draw_convex_hull(mask,landmarks[NOSE_POINTS + MOUTH_POINTS],color=1)
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    mask = (cv2.GaussianBlur(mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    mask = cv2.GaussianBlur(mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return mask

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))


def get_landmarks(im,dlib_rect):
    return np.matrix([[p.x, p.y] for p in predictor(im, dlib_rect).parts()])


class FaceConvert:
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.threshold =[0.6,0.7,0.8]
        self.factor = 0.85
        self.minsize = 20
        with self.graph.as_default():
            self.pnet, self.rnet, self.onet = create_mtcnn(self.sess,None)


    def detect_face(self,img):
        boxes , _  = detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        return boxes


    def swap_face(self,img1,img2):
        boxes = self.detect_face(img1)
        if len(boxes) > 0:
            rect1 = dlib.rectangle(int(boxes[0][0]),int(boxes[0][1]),int(boxes[0][2]),int(boxes[0][3]))
        else:
            raise NoFaces

        boxes = self.detect_face(img2)
        if len(boxes) > 0:
            rect2 = dlib.rectangle(int(boxes[0][0]),int(boxes[0][1]),int(boxes[0][2]),int(boxes[0][3]))
        else:
            raise NoFaces

        landmarks1 = get_landmarks(img1,rect1)
        landmarks2 = get_landmarks(img2,rect2)

        M = transformation_from_points(landmarks1[ALIGN_POINTS],landmarks2[ALIGN_POINTS])
        mask1 = get_face_mask(img1, landmarks1)
        mask2 = get_face_mask(img2, landmarks2)
        warped_mask = warp_im(mask2, M, img1.shape)
        combined_mask = np.max([mask1, warped_mask], axis=0)
        warped_im2 = warp_im(img2, M, img1.shape)
        warped_corrected_im2 = correct_colours(img1, warped_im2, landmarks1)
        output_im = img1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        return output_im

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


# class FaceCompare:
#     def __init__(self):
#         model_exp = os.path.expanduser(model)
#         saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
#         saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))