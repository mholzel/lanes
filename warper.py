import cv2, numpy, os, undistorter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


class Warper():
    def __init__(self, width=1280, height=720):
        self.shape = (width, height)
        h = 0.635 * height
        w = 0.4375 * width
        tl = (w, h)
        bl = (0, height)
        br = (width, height)
        tr = (width - w, h)
        self.src = numpy.float32([tl, bl, br, tr])
        self.dst = numpy.float32([[0, 0],
                                  [0, height],
                                  [width, height],
                                  [width, 0]])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, image):
        return cv2.warpPerspective(image, self.M, self.shape)

    def unwarp(self, image):
        return cv2.warpPerspective(image, self.Minv, self.shape)


def showWarpCoordinates(file='output_images/test_images_undistorted/straight_lines1.jpg', display=False):
    warper = Warper()
    image = mpimg.imread(file)
    cv2.polylines(image, numpy.int32([warper.src]), True, (0, 156, 253), thickness=5)
    off = 20
    cv2.arrowedLine(image, tuple(warper.src[0]), (off, off), (156, 253, 0), thickness=5, tipLength=.05)
    cv2.arrowedLine(image, tuple(warper.src[-1]), (image.shape[1] - off, off), (156, 253, 0), thickness=5,
                    tipLength=.05)
    if display:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    output = 'output_images/straight_lines1_warpcoordinates.jpg'
    Image.fromarray(image).save(output)


def test(dir='test_images', imagetypes=['.jpg', '.png']):
    outputDir = os.path.join('output_images', dir + '_warped/')
    undister = undistorter.Undistorter()
    warper = Warper()
    both = False
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for rawfile in os.listdir(dir):
        file = os.path.join(dir, rawfile)
        if not os.path.isdir(file) and any((file.endswith(type) for type in imagetypes)):
            print(file)
            distorted = mpimg.imread(file)
            undistorted = undister.undistort(distorted)
            warped = warper.warp(undistorted)
            output = os.path.join(outputDir, rawfile)
            if both:
                ax1.imshow(distorted)
                ax1.axis("off")
                ax2.imshow(warped)
                ax2.axis("off")
                plt.savefig(os.path.join(outputDir, rawfile))
            else:
                Image.fromarray(warped).save(output)


if __name__ == "__main__":
    if True:
        showWarpCoordinates()
    else:
        test()
