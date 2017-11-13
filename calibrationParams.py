import cv2, os, pickle, points


def get(width, height, objpoints=None, imgpoints=None):
    '''
    :return: ret, mtx, dist, rvecs, tvecs
    '''
    if objpoints is None or imgpoints is None:
        file = 'calibParams.p'
        if os.path.isfile(file):
            print('Loading pickled calibration parameters')
            return pickle.load(open(file, "rb"))
        else:
            print('Loading points via points.get()')
            objpoints, imgpoints = points.get()
    return cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)


if __name__ == "__main__":
    params = get(width=1280, height=720)
    print(params)
    pickle.dump(params, open('calibParams.p', "wb"))
