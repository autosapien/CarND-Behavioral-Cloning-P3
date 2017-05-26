import cv2


def process_camera_image(image):

    # normalize and mean center to zero
    processed_image = (image / 255.) - 0.5

    # region of interest
    processed_image = processed_image[50:-20, 25:-25]
    #processed_image = processed_image[62:-23, 25:-25]
    #processed_image = processed_image[57:-23, 25:-25]
    #processed_image = processed_image[60:-20, 25:-25]

    # resize
    #processed_image = cv2.resize(processed_image, (200, 66))

    return processed_image

