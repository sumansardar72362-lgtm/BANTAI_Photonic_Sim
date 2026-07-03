import cv2
import numpy as np
import os
from kecia_nn import KeciaLinearLayer

print("\n=======================================================")
print(" 👁️ KECIA VISION SYSTEM - REAL IMAGE PROCESSING")
print("=======================================================\n")

# 1. Amra ekti image file-er nam set korchi
image_name = "user_video_frame.jpg"

# 2. Jodi folder-e kono image na thake, tahole eii code-ti nijestei ekti test image baniye nebe
if not os.path.exists(image_name):
    print(f"[Vision] '{image_name}' khunje paoa jayni. Tai ekti dummy frame toiri kora hochhe...")
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Ekti shada box toiri kora hochhe jeta amader "Product" hisebe kaj korbe
    cv2.rectangle(dummy_img, (50, 50), (250, 250), (255, 255, 255), -1)
    cv2.imwrite(image_name, dummy_img)
    print(f"[Vision] Test image '{image_name}' successfully generated!\n")

# 3. OpenCV diye real image-ti read kora hochhe
print(f"[Vision] Loading real image: {image_name}")
img = cv2.imread(image_name)

# 4. Image-ke choto kore 28x28 size-e ana (Mone achhe to? Total 784 pixels lagbe)
resized_img = cv2.resize(img, (28, 28))

# 5. Image-ke color theke black & white (Grayscale)-e convert kora
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# 6. Pixel value-gulo ke 0 theke 1 er bhetor matrix array-te convert kora (Normalization)
flat_pixels = gray_img.flatten() / 255.0
print(f"[Vision] Image successfully matrixed into {len(flat_pixels)} real pixel values.")

# 7. Kecia AI Vision Layer initialize kora (784 inputs -> 3 outputs: Person, Product, Background)
vision_layer = KeciaLinearLayer(input_features=784, output_features=3)

# 8. Real pixel data fotor fotor laser-er bhetor fire kora
print("\n[Vision] Sending real-world image matrix to Photonic Core...")
detection_result = vision_layer.forward(flat_pixels)

print("\n=======================================================")
print(" ✅ REAL IMAGE PROCESSED SUCCESSFULLY BY KECIA ENGINE!")
print("=======================================================\n")