import CS373LicensePlateDetection
import easyocr
from PIL import Image
from numpy import asarray

# read given image
image = Image.open("numberplate1.png")

# get license plate box position - same method as CS373LicensePlateDetection
(image_width, image_height, px_array_r, px_array_g,
 px_array_b) = CS373LicensePlateDetection.readRGBImageToSeparatePixelArrays("numberplate1.png")
boxPosition = CS373LicensePlateDetection.detectPlate(px_array_r, px_array_g, px_array_b, image_width, image_height)

bbox_min_x = boxPosition[0]
bbox_max_x = boxPosition[1]
bbox_min_y = boxPosition[2]
bbox_max_y = boxPosition[3]

# crop read image to just show the license plate
cropped_plate = image.crop((bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y))

numpy_plate = asarray(cropped_plate)

# read the license plate numbers and letters using easyocr
reader = easyocr.Reader(['en'])
result = reader.readtext(numpy_plate)
print("License Plate: " + result[0][1])
