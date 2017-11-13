import cv2, math, numpy, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider
from undistorter import Undistorter
from warper import Warper
from PIL import Image

old = False


def defaultSobelThresh():
    if old:
        return (65, 255)
    else:
        return (30, 255)


def defaultSobelKsize():
    if old:
        return 7
    else:
        return 7


def sobel(image, orient='x', thresh=None, ksize=None):
    '''
    Converts an RGB image to grayscale, then applies either the
    x or y axis Sobel operator if orient='x' or 'y'. If orient takes
    neither of these values (for example, the default value is None),
    then this computes the magnitude of the x and y-axis operators.
    After this, the points are then scaled to the range [0,255],
    and then points in the specified range are returned in a mask of bools.
    '''
    if thresh is None:
        thresh = defaultSobelThresh()
    if ksize is None:
        ksize = defaultSobelKsize()
    if image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    if orient is 'x':
        abs = numpy.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    elif orient is 'y':
        abs = numpy.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
    else:
        x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        abs = numpy.sqrt(x ** 2 + y ** 2)
    scaled = numpy.uint8(255 * abs / numpy.max(abs))
    return ((scaled >= min(thresh)) & (scaled <= max(thresh)))


def defaultHLSThresh():
    if old:
        return (215, 255)
    else:
        return (150, 255)


def hls(image, thresh=None):
    '''
    Converts an RGB image to hls, then returns
    a mask indicating which pixels are in the specified range
    '''
    if thresh is None:
        thresh = defaultHLSThresh()
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 2]
    return (hls > min(thresh)) & (hls < max(thresh))


def defaultGrayThresh():
    return (.5, 1.)


def grayFilter(image, thresh=None):
    '''
    Only keep the specified percentile threshold (between 0 and 1) of pixels
    '''
    if thresh is None:
        thresh = defaultGrayThresh()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lower = numpy.percentile(gray, 100 * min(thresh))
    upper = numpy.percentile(gray, 100 * max(thresh))
    return (gray >= lower) & (gray <= upper)


def mask(image, orient='x', sobelThresh=None, sobelKsize=None, hlsThresh=None, grayThresh=None):
    sobelled = sobel(image, orient=orient, thresh=sobelThresh, ksize=sobelKsize)
    hlsled = hls(image, thresh=hlsThresh)
    grayed = grayFilter(image, thresh=grayThresh)
    if old:
        return sobelled | hlsled
    else:
        return (sobelled | hlsled) & grayed


def tune(dir='test_images', imagetypes=['.jpg', '.png'], orient=None):
    # Unwarp the images
    undister = Undistorter()
    warper = Warper()
    warped = []
    for rawfile in os.listdir(dir):
        file = os.path.join(dir, rawfile)
        if not os.path.isdir(file) and any((file.endswith(type) for type in imagetypes)):
            print(file)
            distorted = mpimg.imread(file)
            undistorted = undister.undistort(distorted)
            warped.append(warper.warp(undistorted))

    # Create the figure and subplots that will hold the images
    cols = math.floor(math.sqrt(len(warped)))
    cols = 3
    rows = math.ceil(len(warped) / cols)
    fig, ax = plt.subplots(rows, cols)
    axes = ax.ravel()
    for ax in axes:
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=.3, top=1, wspace=0.01, hspace=0.01)

    # Define the default values
    sobelThresh = defaultSobelThresh()
    sobelKsize = defaultSobelKsize()
    hlsThresh = defaultHLSThresh()

    # Now apply the filters to all of the images and show them
    targets = []
    for ax, image in zip(axes, warped):
        sobelled = sobel(image, orient=orient, thresh=sobelThresh, ksize=sobelKsize)
        hlsled = hls(image, thresh=hlsThresh)
        targets.append(ax.imshow(sobelled | hlsled, cmap='gray'))

    # Create the slider objects
    slideraxes = []
    for x in range(1, 6):
        left = .1
        height = .03
        slideraxes.append(plt.axes([left, x * height, 1 - left - .05, height]))
    sobelMinSlider = Slider(slideraxes[0], 'sobel min', 0, 255, valinit=min(sobelThresh))
    sobelMaxSlider = Slider(slideraxes[1], 'sobel max', 0, 255, valinit=max(sobelThresh))
    sobelKerSlider = Slider(slideraxes[2], 'sobel ksize', 3, 15, valinit=sobelKsize)
    hlsMinSlider = Slider(slideraxes[3], 'hls min', 0, 255, valinit=min(hlsThresh))
    hlsMaxSlider = Slider(slideraxes[4], 'hls max', 0, 255, valinit=max(hlsThresh))

    def round_to_odd(f):
        return int(numpy.round((f - 1) / 2) * 2 + 1)

    # Create the callbacks to redraw when one of the values changes
    def update(val):
        print("Updating:", val)
        sobelThresh = (sobelMinSlider.val, sobelMaxSlider.val)
        sobelKsize = round_to_odd(sobelKerSlider.val)
        hlsThresh = (hlsMinSlider.val, hlsMaxSlider.val)
        for ax, image, target in zip(axes, warped, targets):
            sobelled = sobel(image, orient=orient, thresh=sobelThresh, ksize=sobelKsize)
            hlsled = hls(image, thresh=hlsThresh)
            target.set_data(sobelled | hlsled)
        fig.canvas.draw_idle()

    sobelMinSlider.on_changed(update)
    sobelMaxSlider.on_changed(update)
    sobelKerSlider.on_changed(update)
    hlsMinSlider.on_changed(update)
    hlsMaxSlider.on_changed(update)
    plt.show()


def save(dir='project_video', imagetypes=['.jpg', '.png']):
    outputDir = os.path.join('output_images', dir + '_masked/')
    undister = Undistorter()
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
            masked = numpy.uint8(255) * mask(warped)
            output = os.path.join(outputDir, rawfile)
            Image.fromarray(masked).save(output)


if __name__ == "__main__":
    # tune(orient='x')
    save()
