import cv2
import numpy as np
from matplotlib import pyplot as plt

# Tải ảnh
image = cv2.imread('phongcanh.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Ảnh âm tính
negative_image = 255 - image

# 2. Tăng cường độ tương phản (Sử dụng cân bằng histogram)
contrast_image = cv2.equalizeHist(image)

# 3. Biến đổi log
c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1))

# Chuẩn hóa ảnh sau biến đổi log
log_image = np.array(log_image, dtype=np.uint8)

# 4. Cân bằng Histogram
hist_eq_image = cv2.equalizeHist(image)

# Hiển thị các ảnh đã xử lý (bỏ ảnh gốc)
titles = ['Ảnh âm tính', 'Tăng cường độ tương phản', 'Biến đổi Log', 'Cân bằng Histogram']
images = [negative_image, contrast_image, log_image, hist_eq_image]

for i in range(4):
    plt.subplot(2, 2, i+1)  # Chỉnh layout chỉ hiển thị 4 ảnh đã xử lý
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

# Lưu ảnh đầu ra
cv2.imwrite('negative_image.jpg', negative_image)
cv2.imwrite('contrast_image.jpg', contrast_image)
cv2.imwrite('log_image.jpg', log_image)
cv2.imwrite('hist_eq_image.jpg', hist_eq_image)
