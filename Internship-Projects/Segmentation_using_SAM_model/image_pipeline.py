import cv2
import numpy as np

points = []
labels = []
box = []
drawing = False
ix, iy = -1, -1
image_display = None

def mouse_handler(event, x, y, flags, param):
    global points, labels, box, drawing, ix, iy, image_display

    # LEFT: draw box (drag)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        box = []

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = image_display.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (255, 0, 0), 2)
            cv2.imshow("Annotate", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        box = [x1, y1, x2, y2]
        cv2.rectangle(image_display, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # RIGHT: background point
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)
        cv2.circle(image_display, (x, y), 5, (0, 0, 255), -1)

    # MIDDLE (or SHIFT+click alternative): foreground point
    elif event == cv2.EVENT_MBUTTONDOWN:
        points.append([x, y])
        labels.append(1)
        cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Annotate", image_display)


def segment_with_box_and_points(predictor, image_path):
    global points, labels, box, image_display

    points, labels, box = [], [], []

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    image_display = image.copy()
    cv2.imshow("Annotate", image_display)
    cv2.setMouseCallback("Annotate", mouse_handler)

    print("Drag LEFT to draw box | MIDDLE click = FG | RIGHT click = BG")
    print("Press ENTER to run | ESC to cancel")

    while True:
        cv2.imshow("Annotate", image_display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            break
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return image, np.zeros(image.shape[:2], dtype=bool)

    if len(box) != 4:
        print("Draw a box first.")
        cv2.destroyAllWindows()
        return image, np.zeros(image.shape[:2], dtype=bool)

    input_box = np.array([box])
    input_point = np.array(points) if len(points) else None
    input_label = np.array(labels) if len(labels) else None

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False
    )

    mask = masks[0]

    # light refinement (optional)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    cv2.destroyAllWindows()
    return image, mask