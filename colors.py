import cv2, numpy
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
from PIL import Image


def split(image, outputFilename=None, display=True, horizontal=False, size_inches=(15, 7), title=None):
    height, width, channels = image.shape
    if horizontal:
        dst = numpy.zeros((height, channels * width), dtype=numpy.uint8)
        for channel in range(channels):
            dst[:, channel * width:(channel + 1) * width] = image[:, :, channel]
    else:
        dst = numpy.zeros((channels * height, width), dtype=numpy.uint8)
        for channel in range(channels):
            dst[channel * height:(channel + 1) * height, :] = image[:, :, channel]
    if outputFilename is not None:
        Image.fromarray(dst).save(outputFilename)
    if display:
        fig = plt.figure()
        fig.set_size_inches(size_inches[0], size_inches[1])
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.imshow(dst, cmap='gray', vmin=0, vmax=255)
        if title is not None:
            plt.title(title)
        plt.axis("off")
        plt.show()


def toggle(image, original=None, defaults=None, usePercentiles=None):
    '''
    Create an interactive plot in which the user can toggle the thesholds for each of the image channels.
    '''
    # Create subplots in the figure
    height, width, channels = image.shape
    fig, ax = plt.subplots(2, channels)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=.7, hspace=0, wspace=0)
    # Turn off all axes
    for row in ax:
        for elem in row:
            elem.axis("off")

    # Put in some blank placeholders
    zeros = numpy.zeros(image.shape[:2], dtype=bool)
    targets = []
    for channel in range(channels):
        axis = ax[0][channel]
        targets.append(axis.imshow(zeros, cmap='gray', vmin=0, vmax=1))
    targets.append(ax[1][1].imshow(zeros, cmap='gray', vmin=0, vmax=1))

    # If provided, also show the original
    if original is not None:
        ax[1][0].imshow(original)

    # Create the defaults if none were specified
    if defaults is None:
        minmin, minmax, minval = 0, 255, 100
        maxmin, maxmax, maxval = 0, 255, 200
    # A value indicating whether to
    #  0: Use the "or" operator when adding this image
    #  1: Use the "and" operator wen adding this image
    #  2: Do not include this image in the result
    useOr = True
    useOn = True
    defaults = []
    for ch in range(channels):
        minmin, minmax = numpy.min(image[:, :, ch]), numpy.max(image[:, :, ch])
        minmin, minmax = 0, 255
        minval = minmin + .2 * (minmax - minmin)
        maxmin, maxmax = minmin, minmax
        maxval = maxmin + .8 * (maxmax - maxmin)
        defaults.append([minmin, minmax, minval, maxmin, maxmax, maxval, useOr, useOn])

    # Define the update method
    minSliders = []
    maxSliders = []
    checkOnOff = []
    checkAndOr = []

    def isChecked(checkBox):
        return checkBox.lines[0][0].get_visible()

    def update(channel, val):

        # Update each of the individual channels
        images = []
        for ch in range(channels - 1):
            c = image[:, :, ch]
            im = (c >= minSliders[ch].val) & (c <= maxSliders[ch].val)
            images.append(im)
            targets[ch].set_data(im)

        # For the last channel, we remove the specified percent of pixels
        ch = channels - 1
        c = image[:, :, ch]
        lower = numpy.percentile(c, 100 * minSliders[ch].val / 256.)
        upper = numpy.percentile(c, 100 * maxSliders[ch].val / 256.)
        im = (c >= lower) & (c <= upper)
        images.append(im)
        targets[ch].set_data(im)

        # Now combine the images
        combined = numpy.zeros_like(images[0])
        for ch, im in enumerate(images):
            if isChecked(checkOnOff[ch]):
                if isChecked(checkAndOr[ch]):
                    combined = combined & im
                else:
                    combined = combined | im
        targets[-1].set_data(combined)
        fig.canvas.draw_idle()

    # Create min and max sliders for each axis
    top = .95
    sliderOffset = .05
    sliderHeight = 0.03
    sliderWidth = 1 / channels - 2.5 * sliderOffset
    for i, (channel, d) in enumerate(zip(range(channels), defaults)):
        axes = plt.axes([i / channels + sliderOffset, top - sliderHeight, sliderWidth, sliderHeight])
        minSliders.append(Slider(axes, 'min', d[0], d[1], valinit=d[2], valfmt='%0.0f'))
        minSliders[-1].on_changed(lambda x: update(channel, x))

        axes = plt.axes([i / channels + sliderOffset, top - 3 * sliderHeight, sliderWidth, sliderHeight])
        maxSliders.append(Slider(axes, 'max', d[3], d[4], valinit=d[5], valfmt='%0.0f'))
        maxSliders[-1].on_changed(lambda x: update(channel, x))

        axes = plt.axes([i / channels + sliderOffset, top - 6 * sliderHeight, sliderWidth, 2 * sliderHeight])
        checkOnOff.append(CheckButtons(axes, ("On/Off",), (True,)))
        checkOnOff[-1].on_clicked(lambda x: update(channel, x))

        axes = plt.axes([i / channels + sliderOffset, top - 9 * sliderHeight, sliderWidth, 2 * sliderHeight])
        checkAndOr.append(CheckButtons(axes, ("And (checked) / Or (unchecked)",), (True,)))
        checkAndOr[-1].on_clicked(lambda x: update(channel, x))

    update(0, 1)
    plt.show()


if __name__ == "__main__":

    file = 'test_images/test5.jpg'
    file = 'output_images/project_video_warped/frame558.jpg'
    file = 'output_images/project_video_warped/frame866.jpg'
    file = 'output_images/project_video_warped/frame1044.jpg'
    bgr = cv2.imread(file)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gry = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    h, w, ch = rgb.shape

    ksize = 7
    abs = numpy.absolute(cv2.Sobel(gry, cv2.CV_64F, 1, 0, ksize=ksize))
    sobelX = numpy.uint8(255 * abs / numpy.max(abs))

    abs = numpy.absolute(cv2.Sobel(gry, cv2.CV_64F, 0, 1, ksize=ksize))
    sobelY = numpy.uint8(255 * abs / numpy.max(abs))

    abs = numpy.sqrt(sobelX ** 2 + sobelY ** 2)
    sobelXY = numpy.uint8(255 * abs / numpy.max(abs))

    layers = [sobelX, S, gry]
    combined = numpy.zeros((h, w, len(layers)))
    for i, layer in enumerate(layers):
        combined[:, :, i] = layer

    # split(image, horizontal=False)
    # toggle(cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL), original=rgb)
    toggle(combined, original=rgb)
