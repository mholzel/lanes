import cv2, masker, math, movify, numpy, os, sys, time
from undistorter import Undistorter
from warper import Warper
from scipy import optimize
import matplotlib.pyplot as plt


class SingleLaneCost():
    def __init__(self, image, transform=numpy.abs, scalar=True):

        # Save the transform and number of rows
        self.transform = transform
        self.rows = image.shape[0]
        self.scalar = scalar

        # Get the nonzero entries
        r, c = numpy.nonzero(image)
        self.elems = len(r)

        # Now we want to sort these so that we have a dictionary
        # where the keys are the row numbers of rows with nonzero entries,
        # and the values are lists of points in that row that are nonzero.
        # For example, if row 7 has the three nonzero points
        #
        # (7,100), (7,123), (7,998)
        #
        # then the dictionary should return
        #
        # nnz[7] == [100,123,998]
        self.nnz = {}
        for row, col in zip(r, c):
            if row in self.nnz:
                self.nnz[row].append(col)
            else:
                self.nnz[row] = [col]

    def compute(self, poly):
        if self.scalar:
            cost = 0
            for row in self.nnz.keys():

                # Evaluate the polynomial at this row
                x = numpy.polyval(poly, self.rows - row)

                # Now add up the cost
                for val in self.nnz[row]:
                    cost += self.transform(x - val)
        else:
            cost = numpy.zeros(self.elems)
            indx = 0
            for row in self.nnz.keys():

                # Evaluate the polynomial at this row
                x = numpy.polyval(poly, self.rows - row)

                # Now add up the cost
                for val in self.nnz[row]:
                    cost[indx] = self.transform(x - val)
                    indx += 1
        return cost


class CombinedCost():
    def __init__(self, image, transform=numpy.abs, scalar=True):
        # Split the image into left and right
        height, width = image.shape
        left = image[:, :width // 2]
        right = image[:, width // 2:]
        self.width = width

        # Define cost functions for the left and right sides
        self.leftCost = SingleLaneCost(left, transform=transform, scalar=scalar)
        self.rightCost = SingleLaneCost(right, transform=transform, scalar=scalar)
        self.scalar = scalar

    def compute(self, params):
        # The first len(params)-1 parameters define the
        # coefficients of the left lane polynomial, that is,
        #
        # leftLanePoly = params[:-1]
        #
        # The right lane polynomial is obtained by changing the intercept of this polynomial:
        #
        # rightLanePoly = numpy.append( params[:-2], params[-1] )
        #
        # Note that since we define the origin of the coordinate system for the polynomials
        # to be the bottom left hand corner, f(y=0) just takes the value of the final coefficient.
        # Hence the last coefficient defines the intercept.
        p = list(params[:-1])
        cost = self.leftCost.compute(p)
        p[-1] += params[-1] - self.width / 2
        if self.scalar:
            cost += self.rightCost.compute(p)
        else:
            cost = numpy.append(cost, self.rightCost.compute(p))
        return cost


class LaneFinder():
    def __init__(self, x0=None, undistorter=None, warper=None):
        if undistorter is None:
            undistorter = Undistorter()
        if warper is None:
            warper = Warper()
        self.x0 = x0
        self.undistorter = undistorter
        self.warper = warper

    def find(self, image, x0=None, display=False):
        # Create the masked image
        undistorted = self.undistorter.undistort(image)
        warped = self.warper.warp(undistorted)
        masked = masker.mask(warped)
        height, width = masked.shape

        # Set the initial guess
        if x0 is None:
            x0 = self.x0
        if x0 is None:
            x0 = [0, 0, .25 * width, .7 * width]

        # Unfortunately, the best optimizer (bfgs) does not work wth bounds.
        # The optimizers that do (e.g. slsqp and L-BFGS-B) perform horribly.
        # The only other solution appears to be "tnc" which supports bounds, but does not always do well.
        # If you don't mind dropping the bound constraints, you should consider switching to bfgs.
        # If you want to enforce the constraints, then try 'tnc'
        bounds = [(-1, 1), (-width, width), (0, .5 * width), (.6 * width, .8 * width)]

        # Create the cost function and optimize it
        if False:
            cost = CombinedCost(masked)
            params = optimize.minimize(cost.compute, x0, method='bfgs')
        else:
            lb = [b[0] for b in bounds]
            ub = [b[1] for b in bounds]
            costVector = CombinedCost(masked, scalar=False, transform=lambda x: x)
            params = optimize.least_squares(costVector.compute, x0, loss='soft_l1', bounds=(lb, ub))

        # Extract the left and right lanes from the solution
        leftLane = params.x[:-1]
        rightLane = list(leftLane)
        rightLane[-1] += params.x[-1]

        # Now we will overlay this mask on top of the original image
        lanes = numpy.zeros((height, width, 3), dtype=numpy.uint8)
        # Show the left lane in red
        lanes[:, :width // 2, 0] = 255 * masked[:, :width // 2]
        # Show the right lane in green
        lanes[:, width // 2:, 1] = 255 * masked[:, width // 2:]
        # Show the lane in blue
        for y in range(height):
            xL = min(width, max(0, int(numpy.round(numpy.polyval(leftLane, height - y)))))
            xR = min(width, max(0, int(numpy.round(numpy.polyval(rightLane, height - y)))))
            lanes[y, xL:xR, 2] = 255
        # Unwarp the image and overlay on the original
        unwarped = self.warper.unwarp(lanes)
        overlayed = cv2.addWeighted(unwarped, .8, undistorted, 1, 0, numpy.zeros_like(undistorted))

        if display:
            fig,ax = plt.subplots(1,2)
            fig.subplots_adjust(hspace=0, wspace=0, bottom=0, left=0, top=1, right=1)
            fig.set_size_inches(15,7)
            maskedRGB = numpy.zeros_like( lanes )
            masked = numpy.uint8(255) * masked
            maskedRGB[:,:,0] = masked
            overlayedMasked = cv2.addWeighted(lanes, .8, maskedRGB, 1, 0, numpy.zeros_like(maskedRGB))
            ax[0].imshow(overlayedMasked)
            ax[0].axis("off")
            ax[1].imshow(overlayed)
            ax[1].axis("off")
            plt.show()

        # Finally, update the saved parameters
        self.x0 = params.x
        return overlayed


def findLanesinVideo(inputPath, outputPath):
    finder = LaneFinder()
    movify.convert(inputPath, outputPath, finder.find)


if __name__ == "__main__":
    if False:
        file = 'aaa/frame1.jpg'
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        LaneFinder().find(image, display=True)
    else:
        srcDir = "test_videos"
        dstDir = "output_videos"
        for video in sorted(os.listdir(srcDir), reverse=True):
            video = 'project_video.mp4'
            print(video)
            findLanesinVideo(os.path.join(srcDir, video), os.path.join(dstDir, video))
            break
