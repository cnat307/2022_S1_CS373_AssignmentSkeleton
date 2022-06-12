import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i, x in enumerate(pixel_array_r):
        for j, y in enumerate(x):
            value = y * 0.299
            greyscale_pixel_array[i][j] += value

    for i, x in enumerate(pixel_array_g):
        for j, y in enumerate(x):
            value = y * 0.587
            greyscale_pixel_array[i][j] += value

    for i, x in enumerate(pixel_array_b):
        for j, y in enumerate(x):
            value = y * 0.114
            greyscale_pixel_array[i][j] += value

    for i, x in enumerate(greyscale_pixel_array):
        for j, y in enumerate(x):
            greyscale_pixel_array[i][j] = round(greyscale_pixel_array[i][j])

    return greyscale_pixel_array


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    minValue = 255
    maxValue = 0

    for i, x in enumerate(pixel_array):
        if min(x) < minValue:
            minValue = min(x)
        if max(x) > maxValue:
            maxValue = max(x)

    return (minValue, maxValue)


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    minAndMax = computeMinAndMaxValues(pixel_array, image_width, image_height)

    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i, x in enumerate(pixel_array):
        for j, y in enumerate(x):
            if (minAndMax[1] - minAndMax[0]) == 0:
                scale = 0
            else:
                scale = round((y - minAndMax[0]) * (255 / (minAndMax[1] - minAndMax[0])));

            if scale < 0:
                output[i][j] = 0
            elif scale > 255:
                output[i][j] = 255
            else:
                output[i][j] = scale

    return output


def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i, x in enumerate(pixel_array):
        for j, y in enumerate(x):

            if not (i <= 1 or i >= image_height - 2 or j <= 1 or j >= image_width - 2):

                slice = pixel_array[i - 2][j - 2:j + 3], pixel_array[i - 1][j - 2:j + 3], pixel_array[i][j - 2:j + 3], pixel_array[i + 1][j - 2:j + 3], pixel_array[i + 2][j - 2:j + 3]
                sobel = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]

                for k, z in enumerate(slice):
                    for l, a in enumerate(z):
                        slice[k][l] = slice[k][l] * sobel[k][l]

                mean = sum(map(sum, slice)) / 25
                sigma = 0.0
                for b, row in enumerate(slice):
                    for c, col in enumerate(row):
                        sigma += (row[c] - mean) * (row[c] - mean)

                std = math.sqrt(sigma / 25)

                output[i][j] = std

    return output


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    greyscale = createInitializedGreyscalePixelArray(image_width, image_height)

    for i, x in enumerate(pixel_array):
        for j, y in enumerate(x):
            if y >= threshold_value:
                greyscale[i][j] = 1

    return greyscale


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i, x in enumerate(pixel_array):
        for j, y in enumerate(x):
            if not (i < 1 or i > image_height - 2 or j < 1 or j > image_width - 2):

                slice = pixel_array[i - 1][j - 1:j + 2], pixel_array[i][j - 1:j + 2], pixel_array[i + 1][j - 1:j + 2]

                hit = 0
                for a, row in enumerate(slice):
                    for b, col in enumerate(row):
                        if not (col == 0):
                            hit = 1

                if hit == 1:
                    output[i][j] = 1

    return output


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    for i, row in enumerate(pixel_array):
        pixel_array[i].append(0)
        pixel_array[i].insert(0, 0)

    pixel_array.append([0] * (image_width + 2))
    pixel_array.insert(0, [0] * (image_width + 2))

    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i, x in enumerate(pixel_array):
        for j, y in enumerate(x):
            if not (y == 0):
                slice = pixel_array[i - 1][j - 1:j + 2], pixel_array[i][j - 1:j + 2], pixel_array[i + 1][j - 1:j + 2]

                fit = 1
                for a, row in enumerate(slice):
                    for b, col in enumerate(row):
                        if col == 0:
                            fit = 0

                if fit == 1:
                    output[i - 1][j - 1] = 1

    return output

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    currentLabel = 1
    labels = {}
    visited = createInitializedGreyscalePixelArray(image_width, image_height)
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i, x in enumerate(pixel_array):
        for j, y in enumerate(x):

            if not (pixel_array[i][j] == 0) and visited[i][j] == 0:
                visited[i][j] = 1
                labels[currentLabel] = 0

                q = Queue()
                Queue.enqueue(q, [i, j])

                while not (Queue.isEmpty(q)):
                    pixel = Queue.dequeue(q)
                    r = pixel[0]
                    c = pixel[1]

                    output[r][c] = currentLabel
                    visited[r][c] = 1

                    if ((r - 1 >= 0) and visited[r - 1][c] == 0 and not (pixel_array[r - 1][c] == 0)):
                        Queue.enqueue(q, [r - 1, c])
                        output[r - 1][c] = currentLabel
                        visited[r - 1][c] = 1

                    if ((r + 1 <= image_height - 1) and visited[r + 1][c] == 0 and not (pixel_array[r + 1][c] == 0)):
                        Queue.enqueue(q, [r + 1, c])
                        output[r + 1][c] = currentLabel
                        visited[r + 1][c] = 1

                    if ((c - 1 >= 0) and visited[r][c - 1] == 0 and not (pixel_array[r][c - 1] == 0)):
                        Queue.enqueue(q, [r, c - 1])
                        output[r][c - 1] = currentLabel
                        visited[r][c - 1] = 1

                    if ((c + 1 <= image_width - 1) and visited[r][c + 1] == 0 and not (pixel_array[r][c + 1] == 0)):
                        Queue.enqueue(q, [r, c + 1])
                        output[r][c + 1] = currentLabel
                        visited[r][c + 1] = 1

                currentLabel += 1

    for i, x in enumerate(output):
        for j, y in enumerate(x):
            if not (y == 0):
                labels[y] = labels.get(y) + 1

    return output, labels

def createBoundingBox(component_array, component_labels, image_width, image_height):

    total_pixels = 0
    component_index = 0
    for x in component_labels.keys():
        if component_labels[x] > total_pixels:
            component_index = x
            total_pixels = component_labels[x]

    min_x = image_height
    max_x = 0
    min_y = image_width
    max_y = 0
    for i, x in enumerate(component_array):
        for j, y in enumerate(x):
            if y == component_index:

                if i < min_y:
                    min_y = i

                if i > max_y:
                    max_y = i

                if j < min_x:
                    min_x = j

                if j > max_x:
                    max_x = j

    return [min_x, max_x, min_y, max_y]


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate2.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here

    # Step 1:
    # Convert RGB to greyscale image
    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    # Stretch the values to lie between 0 and 255
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    # Step 2:
    # Computing the standard deviation in the 5x5 pixel neighbourhood
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    # Stretch the result to lie between 0 and 255
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    # Step 3: Perform thresholding operation
    threshold_value = 150
    px_array = computeThresholdGE(px_array, threshold_value, image_width, image_height)

    # Step 4: Perform morphological closing with multiple dilation and erosion steps
    for test in range(4):
        px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

    for test in range(4):
        px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

    # Step 5: Finding bounding box

    # Label components of pixel array
    connectedComponents = computeConnectedComponentLabeling(px_array, image_width, image_height)
    component_array = connectedComponents[0]
    component_labels = connectedComponents[1]

    boxPosition = createBoundingBox(component_array, component_labels, image_width, image_height)

    # get x,y min - max values of bounding box
    bbox_min_x = boxPosition[0]
    bbox_max_x = boxPosition[1]
    bbox_min_y = boxPosition[2]
    bbox_max_y = boxPosition[3]

    px_array = px_array_r

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()