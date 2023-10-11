import cv2


def create_mouse_callback(image, list_of_cords):
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            list_of_cords.append((x, y))

            if len(list_of_cords) > 1:
                cv2.line(
                    image,
                    list_of_cords[-2],
                    list_of_cords[-1],
                    (0, 0, 255),
                    thickness=1,
                )

    return onMouse


def main() -> None:
    image = cv2.imread("assets/image.png")
    list_of_cords = []
    window_name = "Window"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        create_mouse_callback(image, list_of_cords),
    )

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            x_cords = [cord[0] for cord in list_of_cords]
            y_cords = [cord[1] for cord in list_of_cords]

            top, bottom = min(y_cords), max(y_cords)
            left, right = min(x_cords), max(x_cords)

            print("Saving image with lines...")
            cv2.imwrite(
                "assets/image_with_lines.png",
                image[top:bottom, left:right],
            )
        elif key == ord("r"):
            image = cv2.imread("assets/image.png")
            list_of_cords = []

            cv2.setMouseCallback(
                window_name,
                create_mouse_callback(image, list_of_cords),
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
