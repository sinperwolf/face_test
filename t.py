import cv2
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from facenet.align import detect_face as df
from face_convert import FaceConvert


# threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
# factor = 0.709 # scale factor
# minsize = 20

img1 = cv2.imread('face1.jpg')
# img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)


img2 = cv2.imread('face2.jpg')
# img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)


cvt = FaceConvert()
output_img = cvt.swap_face(img1,img2).astype(np.uint8)    
output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2RGB)

plt.imshow(output_img)
plt.show()