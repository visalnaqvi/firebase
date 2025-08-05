# from PIL import Image
# import piexif

# image = Image.open("yo.jpg")
# image.save("output.jpg" ,exif=piexif.dump({}), lossless=True)

# from PIL import Image

# image = Image.open("yo.jpg")
# icc_profile = image.info.get("icc_profile")
# image.save("output.jpg" , icc_profile=icc_profile) 

import imageio

img = imageio.imread("yo.jpg")
imageio.imwrite("output.jpg", img)
