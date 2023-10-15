import cv2


def create_mouse_callback(image, list_of_points):
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            list_of_points.append((x, y))
            cv2.circle(
                image,
                list_of_points[-1],
                radius=1,
                color=(0, 0, 255),
                thickness=-1,
            )

            if len(list_of_points) > 1:
                cv2.line(
                    image,
                    list_of_points[-2],
                    list_of_points[-1],
                    (0, 0, 255),
                    thickness=1,
                )

    return onMouse


def save_image(image, list_of_points):
    if len(list_of_points) < 4:
        print("You need at least four points to save the image.")
        return

    x_cords = [cord[0] for cord in list_of_points]
    y_cords = [cord[1] for cord in list_of_points]

    top, bottom = min(y_cords), max(y_cords)
    left, right = min(x_cords), max(x_cords)

    print("Saving image with lines...")
    cv2.imwrite(
        "assets/image_with_lines.png",
        image[top:bottom, left:right],
    )


def reset_state(image_path, window_name):
    image = cv2.imread(image_path)
    list_of_points = []

    cv2.setMouseCallback(
        window_name,
        create_mouse_callback(image, list_of_points),
    )

    return image, list_of_points


def main() -> None:
    """
    Main function of the program.

    q: Quit the program.
    s: Save the image.
    r: Reset the image and list of points.
    """

    window_name = "Window"
    image_path = "assets/image.png"

    cv2.namedWindow(window_name)
    image, list_of_points = reset_state(image_path, window_name)

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            save_image(image, list_of_points)
        elif key == ord("r"):
            image, list_of_points = reset_state(image_path, window_name)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
