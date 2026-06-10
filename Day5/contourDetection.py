# Contour Detection with Detailed Comments
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/input/auscoin.jpg")  # imread = image read

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY  # Convert BGR image to grayscale
)

_, thresh = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
# THRESH_BINARY_INV -> invert black/white
# THRESH_OTSU -> automatically choose best threshold

contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,      # retrieve only outer contours
    cv2.CHAIN_APPROX_SIMPLE # keep only important contour points
)

print("Contours Found =", len(contours))

output = img.copy()
object_count = 0

for contour in contours:

    area = cv2.contourArea(contour)  # area inside contour

    if area < 100:  # ignore noise
        continue

    object_count += 1

    cv2.drawContours(
        output,
        [contour],
        -1,
        (255, 0, 0),  # blue
        4
    )

    x, y, w, h = cv2.boundingRect(contour)
    # x = left, y = top, w = width, h = height

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (255, 0, 255),  # magenta
        2
    )

    cv2.putText(
        output,
        f"C{object_count}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output_rgb)
plt.title(f"Detected Contours = {object_count}")
plt.axis("off")

plt.tight_layout()
plt.show()
