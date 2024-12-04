import cv2

IMAGE_OFFSET = 60

def processImage(image):
    height, width, _ = image.shape

    # Crop to cut off top half and outer 50%
    yStart = height // 2
    yEnd = height
    xStart = width // 4
    xEnd = width * 3 // 4
    croppedImage = image[yStart:yEnd, xStart:xEnd]

    # Crop to square
    minLength = min(yEnd - yStart, xEnd - xStart)

    squareImage = croppedImage[0:minLength, IMAGE_OFFSET:minLength + IMAGE_OFFSET]

    # Downscale to 128x128
    resizedImage = cv2.resize(squareImage, (128, 128))

    # Convert to grayscale
    grayImage = cv2.cvtColor(squareImage, cv2.COLOR_BGR2GRAY)

    # Find edges
    edgeImage = cv2.Canny(grayImage, threshold1=50, threshold2=200)

    return edgeImage