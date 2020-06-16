import cv2

def BGR_to_RGB(img):
    output = img
    output[:, :, 2] = img[:, :, 0]
    output[:, :, 1] = img[:, :, 1]
    output[:, :, 0] = img[:, :, 2]
    return output

img = cv2.imread("weka.jpg")

# BGR -> RGB
img = BGR_to_RGB(img)

# Save result
cv2.imwrite("out.jpg", img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()