import numpy as np
import cv2
import glob
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt
# #############part 1#############
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.005)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:8].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objectpoints = [] # 3d point in real world space
imagepoints = [] # 2d points in image plane.
images = glob.glob('*.JPG')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5,8),None)
   # print(corners()
    # If found, add object points, image points (after refining them)
    if ret == True:
        print('parameters', fname)
        objectpoints.append(objp)
        corners_Subpix = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imagepoints.append(corners_Subpix)
        img = cv2.drawChessboardCorners(img, (5,8), corners_Subpix,ret)
        plt.figure()
        plt.subplot(1,1, 1), plt.imshow(img)
cv2.destroyAllWindows()

# ##########part 2#####################
ret, mtx, distortion, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, gray.shape[::-1], None, None)
print('Camera matrix =  \n' ,mtx)
print('distortion parameters = ' ,distortion)
