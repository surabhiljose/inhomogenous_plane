import numpy as np
import cv2
import glob

# Step 1 : setting the termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Step 2 : Preparing object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Step 3 : Setting arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

checkboardDetectedImages = [] #array to store filenames of all images in which the chessboard was detected.

# Step 4 : reading through all the images(for calibration) and adding image points and and object points
images = glob.glob('*.jpg')
for fname in images:   # loop to read through all the images
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None) # Find the chess board corners
    if ret == True: # If found, add object points, image points (after refining them)
        checkboardDetectedImages.append(fname) #adding image file name for which the chessboard was detected
        objpoints.append(objp) #adding object points
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2) #adding image points

# Step 5 : Using image and object points to find camera matrix, distortion , rotation and translation values of my webcam
try:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print "camera matrix \n"
    print mtx
    print "distortion coefficient \n"
    print dist
    print "rotation vector \n"
    print rvecs
    print "translation vector \n"
    print tvecs
except:
    print "Failed getting cv2.calibrateCamera"
    pass

# Step 6 : Undistorting the first image in which the checkerboard was detected.
img = cv2.imread(checkboardDetectedImages[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h)) #more refined camera matrix to reduce edge blurs
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imwrite("C:\Users\surab\Pictures\mytestpic\undistorted.png", dst) #image saved for later use


#-------------------------HOMOGRAPHY---------------------------------------
# For this part of the assignment :
#   Source Image      = one image(of the checkerboard) taken during calibration
#   Destination Image = undistorted image(of the checkerboard) I had saved earlier.
# I shall project the source image checkerboard in the plane of the checkerborad detected in the destination image.
# Checkerboard corners in both the images shall be used as image points to find the homography

# Step 1 : finding checkerboard corners of the source image
point_src = []
img_src = cv2.imread(checkboardDetectedImages[len(checkboardDetectedImages)-1]) #using the last image in which the checkerboard was detected during calibration.
gray_src = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
ret, corners_src = cv2.findChessboardCorners(gray_src, (7,6),None)
if ret == True:
    point_src = cv2.cornerSubPix(gray_src, corners_src, (11, 11), (-1, -1), criteria) #saving the checkerboard corners


# Step 2 : finding checkerboard corners of the destination image
point_des = []
img_des = cv2.imread("C:\Users\surab\Pictures\mytestpic\undistorted.png")
gray_des = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners_des = cv2.findChessboardCorners(gray_des, (7,6),None)
if ret == True:
    point_des = cv2.cornerSubPix(gray_des, corners_des, (11, 11), (-1, -1), criteria)


# Step 3 : Calculating Homography from source points and destination points.
h, status = cv2.findHomography(np.asarray(point_src), np.asarray(point_des))
print "homography"
print h

# Step 4 : Warping source image to destination based on homography
im_out = cv2.warpPerspective(img_src, h, (img_des.shape[1], img_des.shape[0]))

# Step 5 : Displaying images
cv2.imshow("Source Image", img_src)
cv2.imshow("Destination Image", img_des)
cv2.imshow("Warped Source Image", im_out)
cv2.waitKey(0)

cv2.destroyAllWindows()