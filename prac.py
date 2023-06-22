import cv2
import numpy as np


def remove_background(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Create a mask
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define the background and foreground model
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define the bounding rectangle for the foreground object
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

    # Apply grabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Create a mask where 0 and 2 denote background, while 1 and 3 denote foreground
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


# Example usage
image_path = 'path/to/your/image.jpg'
result_image = remove_background(image_path)

# Display the result
cv2.imshow('Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
