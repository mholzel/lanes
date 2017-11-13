import calibrationParams, cv2, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


class Undistorter():
    def __init__(self, mtx=None, dist=None):
        if mtx is None or dist is None:
            print('Using default camera calibration parameters with width=1280,height=720')
            ret, mtx, dist, rvecs, tvecs = calibrationParams.get(width=1280, height=720)
        self.mtx = mtx
        self.dist = dist

    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)


def test(dir='camera_cal', imagetypes=['.jpg', '.png']):
    outputDir = os.path.join('output_images', dir + '_undistorted/')
    undistorter = Undistorter()
    both = False
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for rawfile in os.listdir(dir):
        file = os.path.join(dir, rawfile)
        if not os.path.isdir(file) and any((file.endswith(type) for type in imagetypes)):
            distorted = mpimg.imread(file)
            undistorted = undistorter.undistort(distorted)
            combined = cv2.hconcat((distorted, undistorted))
            print(distorted.shape, combined.shape)
            grayd = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)
            grayu = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            grayDiff = grayd - grayu
            output = os.path.join(outputDir, rawfile)
            if both:
                ax1.imshow(distorted)
                ax1.axis("off")
                ax2.imshow(grayDiff, cmap='gray', vmin=-255, vmax=255)
                ax2.axis("off")
                plt.savefig(output)
            else:
                Image.fromarray(undistorted).save(output)


if __name__ == "__main__":
    test()
