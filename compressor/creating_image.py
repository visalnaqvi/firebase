from PIL import Image
import numpy as np

# Step 1: Define RGB pixel values (3x3 image = 9 pixels)
# Each pixel is a tuple: (R, G, B)
pixels = [
    (255, 0, 0),   (0, 255, 0),   (0, 0, 255),   # Red, Green, Blue
    (255, 255, 0), (0, 255, 255), (255, 0, 255), # Yellow, Cyan, Magenta
    (255, 255, 255), (128, 128, 128), (0, 0, 0)  # White, Gray, Black
]

# Step 2: Convert to a NumPy array and reshape to image dimensions (3 rows, 3 cols, 3 color channels)
array = np.array(pixels, dtype=np.uint8).reshape((3, 3, 3))

# Step 3: Create image from array
img = Image.fromarray(array, 'RGB')

# Step 4: Save it
img.save("manual_rgb_image.png")
