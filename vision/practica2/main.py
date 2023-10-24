import cv2
import numpy as np


def remove_background(img):
    MASK_COLOR = (0.0, 0.0, 0.0)  # In BGR format

    # == Processing =======================================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100)
    edges = cv2.dilate(edges, None)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(edges.shape)
    for ctn in contours:
        area = cv2.contourArea(ctn)
        if area > 500:
            cv2.fillConvexPoly(mask, ctn, (255))

    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask_stack = np.dstack([mask] * 3).astype("float32") / 255.0
    img = img.astype("float32") / 255.0

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    masked = (masked * 255).astype("uint8")

    return masked


def main():
    window_name = "Window"
    image_path = "assets/persona.jpeg"

    cv2.namedWindow(window_name)

    img = cv2.pyrDown(cv2.imread(image_path))
    img_sol1 = remove_background(img)

    cv2.imshow(window_name, img_sol1)
    cv2.moveWindow(window_name, 2920, 2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
