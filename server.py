# Python Flask server if you want to host this utility on any server

from flask import Flask, request
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage import img_as_float, img_as_ubyte
import cv2
import random
import base64
import os


app = Flask(__name__)

CHILD_SIZE = 20 # the size of the child images in pixel that will be used as collage, childrens will be cropped to make it square
ALPHA = 0.7 # alpha value of the children
WEGHT = 0.5 # how much prominent the color filter should be, the more the value the more it is prominent


@app.route('/', methods = ['POST', 'GET'])
def image():
    if (request.method == "GET"):
        return """
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>image to image</title>
</head>
<body>
<h1>Epsi's Image Collage</h1>
<hr>
<h4>IMPORTANT!! NO GRAY SCALE IMAGE SUPPORTED<h4>
<hr>
<br>
<br>
    <form action="/" method="post" enctype="multipart/form-data">
        <label for="imgSize">Size of the child image in pixel (for ex: 20)</label><br>
        <input type="text" id="imgSize" name="imgSize" required value="20">
        <br>
        <br>
        <label for="weight">Weight of the color filter (for ex: 0.5)</label><br>
        <input type="text" id="weight" name="weight" required value="0.5">
        <br>
        <br>
        <label for="alpha">Alpha value for image superposition (for ex: 0.7) </label><br>
        <input type="text" id="alpha" name="alpha" required value="0.7" required>
        <br>
        <br>
        <label for="master">Select the master image</label>
        <br>
        <input type="file" id="master" name="master" accept="image/*" required>
        <br>
        <br>
        <label for="children">Select multiple child images for collage</label>
        <br>
        <input type="file" id="children" name="children" accept="image/*" multiple>
        <br>
        <br>
        <input type="submit">
    </form>
</body>
</html>
        """

    if (request.method == "POST"):
        try:
            master = None
            children = []

            global CHILD_SIZE, WEGHT, ALPHA

            CHILD_SIZE = int(request.form.get('imgSize', 20))
            WEGHT = float(request.form.get('weight', 0.5))
            ALPHA = float(request.form.get('alpha', 0.7))
            master = request.files["master"]
            children_ = request.files.getlist("children")
            # print(CHILD_SIZE)
            # print(WEGHT)
            # print(ALPHA)
            # print(master)
            # print(children)

            #---------------------->>>>
            print(">>")
            print(type(master))
            filestr = master.read()
            npimg = np.fromstring(filestr, np.uint8)
            master = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            print(f"old master dtype: {master.dtype}")
            master = img_as_float(master)
            print(f"new master dtype: {master.dtype}")
            #cv2.imwrite("master.png", img_as_ubyte(master))
            print(f"master image height: {master.shape[0]} and width: {master.shape[1]}")
            #print(base64.b64encode(master))


            # now we will determine the dominant colors of the master
            master_height = master.shape[0]
            master_width = master.shape[1]

            h_bins = master_width // CHILD_SIZE + (1 if (master_width % CHILD_SIZE > 0) else 0)
            v_bins = master_height // CHILD_SIZE + (1 if (master_height % CHILD_SIZE > 0) else 0)

            print(f"total number of children can be stacked horizontally: {h_bins} and vertically: {v_bins}")

            dominant_colors = np.full((v_bins, h_bins, 3), 1.)

            for child in children_:
                filestr = child.read()
                npimg = np.fromstring(filestr, np.uint8)
                child = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                print(f"old child dtype: {child.dtype}")
                child = img_as_float(child)
                print(f"new child dtype: {child.dtype}")
                #cv2.imwrite("temp1.png", child)
                if (len(child.shape) == 3):
                    child = resize(child, (CHILD_SIZE, CHILD_SIZE))
                    #cv2.imwrite("temp.png", child)
                    children.append(img_as_float(child))
                    print("stored")
                else:
                    print("will not process GrayScale Image")

            h_start = 0
            v_start = 0

            final_image = np.zeros((v_bins * CHILD_SIZE, h_bins * CHILD_SIZE, 3))
            filtered_images = []

            for row in range(v_bins):
                for col in range(h_bins):
                    h_start = col * CHILD_SIZE
                    v_start = row * CHILD_SIZE

                    h = h_start
                    hd = min((h_start + CHILD_SIZE), master_width)

                    v = v_start
                    vd = min((v_start + CHILD_SIZE), master_height)

                    view = master[v:vd, h:hd]
                    # finding dominant color
                    dominant_color = get_dominant_color(view)
                    dominant_colors[row, col] = dominant_color

                    filtered_images.append(apply_color_filter(dominant_color, random.choice(children)))
                    final_image[v:v + CHILD_SIZE, h:h + CHILD_SIZE] = apply_color_filter(dominant_color, random.choice(children))

            v_final_image = final_image.shape[0]
            h_final_image = final_image.shape[1]

            v_master = master.shape[0]
            h_master = master.shape[1]

            final_image = final_image[:v_master, :h_master]

            final_image_with_alpha_background = (1 - ALPHA) * master + ALPHA * final_image

            #---------------------->>>>

            final_image = img_as_ubyte(final_image)
            final_image_with_alpha_background = img_as_ubyte(final_image_with_alpha_background)
            _, im_arr_1 = cv2.imencode('.png', final_image)  # im_arr: image in Numpy one-dim array format.
            im_bytes_1 = im_arr_1.tobytes()
            im_b64_1 = base64.b64encode(im_bytes_1)

            _, im_arr_2 = cv2.imencode('.png', final_image_with_alpha_background)  # im_arr: image in Numpy one-dim array format.
            im_bytes_2 = im_arr_2.tobytes()
            im_b64_2 = base64.b64encode(im_bytes_2)

            return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>image to image</title>
</head>
<body>
<h1>Epsi's Image Collage</h1>
<hr>
<br>
<div>
  <p>With out background superposition</p>
  <img src="data:image/png;base64, {im_b64_1.decode('ascii')}" alt="Red dot" />
</div>
<div>
  <p>With background superposition</p>
  <img src="data:image/png;base64, {im_b64_2.decode('ascii')}" alt="Red dot" />
</div>
</body>
</html>
            """
        except Exception as e:
            print(e)
            return "oops! something went wrong"

# method to get dominant color
def get_dominant_color(img):
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    return dominant


def apply_color_filter(color, image):
    # create an image with a single color (here: red)
    color_img = np.full((image.shape[0], image.shape[1], 3), color, np.float64)

    # add the filter  with a weight factor of 20% to the target image
    filtered_img = cv2.addWeighted(image, (1 - WEGHT), color_img, WEGHT, 0)
    return img_as_float(filtered_img)

if __name__ == '__main__':
    app.run(threaded=True)
