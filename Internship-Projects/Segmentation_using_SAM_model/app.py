from sam_model import load_sam_model
from image_pipeline import segment_with_box_and_points
import cv2

predictor, device = load_sam_model()
print(f"Model on: {device}")

image, mask = segment_with_box_and_points(predictor, "data/test.jpg")

overlay = image.copy()
overlay[mask] = [0, 255, 0]
output = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

cv2.imshow("Segmented Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()