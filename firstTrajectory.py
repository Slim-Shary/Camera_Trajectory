import numpy as np
import cv2
import PyGnuplot as py
import glob
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
# ------------------------calibration from assignment 3--------------------------
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.005)
objp = np.zeros((5*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:5, 0:8].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objectpoints = []  # 3d point in real world space
imagepoints = []  # 2d points in image plane.
images = glob.glob('*.JPG')
for fname in images:
    img = cv2.imread(fname)
    dim = (1280, 720)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5, 8), None)
    # If found, add object points, image points (after refining them)
    if ret is True:
        print('parameters', fname)
        objectpoints.append(objp)
        corners_Subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imagepoints.append(corners_Subpix)
        # img = cv2.drawChessboardCorners(img, (5, 8), corners_Subpix, ret)
        # plt.figure()
        # plt.subplot(1, 1, 1), plt.imshow(img)
cv2.destroyAllWindows()
ret, M, distortion, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, gray.shape[::-1], None, None)
# undistort
newM, roi = cv2.getOptimalNewCameraMatrix(M, distortion, (1280, 720), 1)
print('Camera matrix =  \n', M, "\n")
print('distortion parameters = ', distortion, "\n")
inv = np.linalg.inv(M)
print("inverse \n= ", inv, "\n")
transposed_M = cv2.transpose(inv)
print("transpose \n= ", transposed_M, "\n")
r_temp = np.ones((3, 3))
t_temp = np.zeros((3, 1))
coordinates = np.zeros((3, 1))
array_t = []
arr_x = np.empty(261)
arr_y = np.empty(261)
arr_z = np.empty(261)
arrX = np.empty(261)
arrY = np.empty(261)
arrZ = np.empty(261)
# -----------------------------------trajectory-------------------------------
# Get the video from directory
cap = cv2.VideoCapture('cs9645-assign-4.mov')
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = cv2.undistort(old_frame, M, distortion, None, newM)
cnt = 0
while cap.isOpened():
    cnt = cnt + 1
    print(cnt)
    ret, frame = cap.read()
    if ret is True:
        # undistort
        frame = cv2.undistort(frame, M, distortion, None, newM)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(old_frame, None)
        kp2, des2 = sift.detectAndCompute(frame, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        print("\n Fundamental Matrix = \n", F)
        dot = np.multiply(transposed_M, F)
        E = np.multiply(dot, inv)
        print("\n Essential matrix = \n", E, "\n")
        R1, R2, T = cv2.decomposeEssentialMat(E)
        print("\n Translation = ", T, "\n")
        print("\n Rotation = ", R1, "\n")
        t_temp = T + t_temp
        arr_x[cnt-1] = t_temp[0]
        arr_y[cnt-1] = t_temp[1]
        arr_z[cnt-1] = t_temp[2]
        r_temp = np.multiply(R1, r_temp)
        r_temp_transpose = cv2.transpose(r_temp)
        r_neg = -r_temp_transpose
        C = np.dot(r_neg, t_temp)
        arrX[cnt-1] = C[0]
        arrY[cnt-1] = C[1]
        arrZ[cnt-1] = C[2]
        # Now update the previous frame and previous points
        old_frame = frame.copy()
    else:
        break

# py.s([arr_x, arr_y, arr_z], filename='translation.dat')
# py.c('set title "translation"; set xlabel "x-axis"; set ylabel "y-axis"; set zlabel "z-axis"')
# py.c('splot "translation.dat" using 1:2:3 with lines ')

py.s([arrX, arrY, arrZ], filename='camera.dat')
py.c('set title "camera trajectory"; set xlabel "x-axis"; set ylabel "y-axis"; set zlabel "z-axis"')
py.c('splot "camera.dat" using 1:2:3 with lines ')

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(arr_x, arr_y, arr_z)

plt.show()








