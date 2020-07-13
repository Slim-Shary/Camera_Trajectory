import numpy as np
import cv2
import PyGnuplot as py
import glob
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
# ------------------------calibration from assignment 3-------------------------------------------------------------
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
        img = cv2.drawChessboardCorners(img, (5, 8), corners_Subpix, ret)
cv2.destroyAllWindows()
ret, M, distortion, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, gray.shape[::-1], None, None)
# undistort
newM, roi = cv2.getOptimalNewCameraMatrix(M, distortion, (1280, 720), 1)
print('Camera matrix =  \n', M, "\n")
print('distortion coefficients = ', distortion, "\n")
invM = np.linalg.inv(M)
# -----------------------------------trajectory----------------------------------------------------------------------
r_temp = np.ones((3, 3))
t_temp = np.zeros((3, 1))
coordinates = np.zeros((3, 1))
arr_x = np.empty(261)
arr_y = np.empty(261)
arr_z = np.empty(261)
arrX = np.empty(261)
arrY = np.empty(261)
arrZ = np.empty(261)
# Get the video from directory
cap = cv2.VideoCapture('cs9645-assign-4.mov')
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = cv2.undistort(old_frame, M, distortion, None, newM)
x,y,w,h = roi
old_frame = old_frame[y:y+h, x:x+w]
cnt = 0
while cap.isOpened():
    cnt = cnt + 1
    print("frame", cnt+1, "is read")
    ret, frame = cap.read()
    if ret is True:
        # undistort
        frame = cv2.undistort(frame, M, distortion, None, newM)
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(old_frame, None)
        kp2, des2 = sift.detectAndCompute(frame, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if 0.3*m.distance < 0.8 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                matchesMask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)
        print("\n Fundamental Matrix = \n", F)
        E = invM.T.dot(F).dot(invM)
        print("\n Essential matrix = \n", E, "\n")
        R1, R2, matT = cv2.decomposeEssentialMat(E)
        print("\n decompose Translation = ", matT, "\n")
        print("\n decompose R1 = ", R1, "\n")
        print("\n decompose R2 = ", R2, "\n")
        points, Rot, tran, maskkk = cv2.recoverPose(E, pts1, pts2, M)

        def in_front_of_both_cameras(first_points, second_points, rot, trans):
            # check if the point correspondences are in front of both images
            for first, second in zip(first_points, second_points):
                first_z = np.dot(rot[0, :] - second[0] * rot[2, :], trans) / np.dot(rot[0, :] - second[0] * rot[2, :],
                                                                                    second)
                first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
                second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

                if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                    return False

            return True

        U, S, Vt = np.linalg.svd(E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
        first_inliers = []
        second_inliers = []
        for i in range(len(mask)):
            if mask[i]:
                # normalize and homogenize the image coordinates
                first_inliers.append(invM.dot([pts1[i][0], pts1[i][1], 1.0]))
                second_inliers.append(invM.dot([pts2[i][0], pts2[i][1], 1.0]))

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in front of both cameras
        # First choice: R = U * Wt * Vt, T = +u_3
        R = U.dot(W).dot(Vt)
        T = U[:, 2]
        if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
            print("ikinci")
            # Second choice: R = U * W * Vt, T = -u_3
            T = - U[:, 2]
            if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
                print("ucuncu")
                # Third choice: R = U * Wt * Vt, T = u_3
                R = U.dot(W.T).dot(Vt)
                T = U[:, 2]

                if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
                    print("dorduncu")
                    # Fourth choice: R = U * Wt * Vt, T = -u_3
                    T = - U[:, 2]
        print("\n SVD T = ", T, "\n")
        print("\n SVD R = ", R, "\n")
        print("\n recover Translation = ", tran, "\n")
        print("\n recover Rotation = ", Rot, "\n")
        if cnt == 1:
            r_temp = Rot
            t_temp = tran
            arr_x[cnt - 1] = t_temp[0]
            arr_y[cnt - 1] = t_temp[1]
            arr_z[cnt - 1] = t_temp[2]
            C = np.dot(-r_temp.T, t_temp)
            coordinates = C
            arrX[cnt - 1] = coordinates[0]
            arrY[cnt - 1] = coordinates[1]
            arrZ[cnt - 1] = coordinates[2]
            continue
        r_temp = Rot
        t_temp = t_temp + tran
        arr_x[cnt-1] = t_temp[0]
        arr_y[cnt-1] = t_temp[1]
        arr_z[cnt-1] = t_temp[2]
        C = np.dot(-r_temp.T, t_temp)
        coordinates = coordinates + C
        print(C)
        arrX[cnt-1] = coordinates[0]
        arrY[cnt-1] = coordinates[1]
        arrZ[cnt-1] = coordinates[2]
        # Now update the previous frame and previous points
        old_frame = frame.copy()
    else:
        break

py.s([arrX, arrY, arrZ], filename='camera.dat')
py.c('set title "camera trajectory"; set xlabel "x-axis"; set ylabel "y-axis"; set zlabel "z-axis"')
py.c('splot "camera.dat" using 1:2:3 ')












