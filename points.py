# The basic logic of this code was copied from
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import cv2, numpy, os


def get(dir='camera_cal', shape=(6, 9), imagetypes=['.jpg', '.png'], display=False, verbose=False):
    '''

    :param dir: the directory where to look for the specified imagetypes
    :param shape: the shape of the checkerboard, that is, how many points in the
    (vertical,horizontal) direction we should expect
    :param imagetypes: the filetypes that should be read as images
    :param display: a boolean indicating whether to plot the detected points
    :return: (objpoints, imgpoints), where
    objpoints are 3D points in real-world space
    and
    imgpoints are 2D points in image space
    '''
    # Prepare the object points. If the corners are detected in an image,
    # this list is added to the list of object points. These are 3D points
    # indicating the locations of the corners. This list if of the form
    # [(0,0,0), (1,0,0), (2,0,0) ....,(shape[0],shape[1],0)
    objp = numpy.zeros((shape[0] * shape[1], 3), numpy.float32)
    objp[:, :2] = numpy.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Look for all of the images in the specified directory
    for file in os.listdir(dir):

        file = os.path.join(dir, file)

        # Only try to read the file as an image if
        # 1. it is not a directory
        # AND
        # 2. it is one of the specified imagetypes
        if not os.path.isdir(file) and any((file.endswith(type) for type in imagetypes)):

            # Try reading the image, then convert it to grayscale
            color = cv2.imread(file)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, shape, None)

            # If the corners were found,
            # then add both the 2D and 3D points to the respective lists
            if ret == True:

                imgpoints.append(corners)
                objpoints.append(objp)

                if display:
                    img = cv2.drawChessboardCorners(color, (7, 6), corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(1000)
            elif verbose:
                print("Could not find all corners in", file)

    return objpoints, imgpoints


if __name__ == "__main__":
    get(display=True)
