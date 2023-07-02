import re
import cv2
import numpy as np
import gradio as gr

blocks = gr.Blocks()

# replace multiple spaces with single space
# Sometimes you want to copy command from other terminal result and paste it to your terminal but \n or multiple spaces causes problem...                         
def replace_multiple_spaces_with_single_space(string:str) -> str:
    return (re.sub(r'\s+', ' ', string, flags=re.I)) 

import PIL
from PIL import Image
import numpy as np

# concat images horizontally, useful for combining poses.

def concat_images_horizontally(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    return new_im

def concat_from_files_horizontally(files):
    images = []
    for file in files:
        images.append(Image.open(file))
    return concat_images_horizontally(images)

import glob


def concat_from_glob_horizontally(glob_pattern:str) -> Image.Image:
    """
    Concat images horizontally from glob pattern (directory path)
    ex) concat_from_glob_horizontally("/home/user/*.png")
    """
    files = glob.glob(glob_pattern)
    return concat_from_files_horizontally(files)

# using base64 to decode string

import base64

def decode_base64_to_string(base64_string):
    return base64.b64decode(base64_string).decode('utf-8')

#def autogen_mask_blur(pose_image, radius = 5, threshold = 40):
    # convert to grayscale. Then if value is less than 40, set to 0, otherwise set to 255
    # Then for each white pixel, increase brightness for circle with radius to new image
    # Then apply gaussian blur to new image
    # apply threshold again
    # apply connected components to get separate objects
    # apply different color for each object

def autogen_mask_blur(pose_image, radius=5, threshold=40, gaussian=5):
    """
    From OpenPose, generate mask for each pose.
    """
    # convert to grayscale
    gray = cv2.cvtColor(pose_image, cv2.COLOR_BGR2GRAY)

    # threshold to get a mask
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # create a new image with the same size as the input image
    new_image = np.zeros_like(gray)
    # for each white pixel in the mask, increase brightness for circle with radius in new image
    indices = np.where(mask == 255)
    for z in zip(*indices):
        y, x = z
        cv2.circle(new_image, (x, y), radius, (255, 255, 255), -1)

    # apply gaussian blur to new image
    blurred = cv2.GaussianBlur(new_image, (gaussian, gaussian), 0)
    if debug_image_output is not None:
        debug_image_output.value = blurred

    # threshold to get a mask
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    # assert mask is 1 channel
    assert mask.shape == (pose_image.shape[0], pose_image.shape[1]), "mask should have 1 channel"

    # separate objects with connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # apply different color for each object
    # now use 3 channels
    empty = np.zeros_like(pose_image)
    # randomly generate colors
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    # set background to white
    colors[0] = (255, 255, 255)
    # for each object, set color to the object
    for i in range(1, num_labels):
        empty[labels == i] = colors[i]

    return Image.fromarray(empty), Image.fromarray(new_image), blurred
with blocks:
    with gr.Tabs():
        # first tab is for image concat from directory(glob pattern)
        # second tab is for encoding / decoding base64 
        # third tab is for mask generation
        with gr.TabItem("Image Concat"):
            with gr.Row():
                concat_glob_input = gr.Textbox(lines=1, label="Glob Pattern")
                concat_glob_output = gr.Image(type="pil")
            concat_glob_button = gr.Button("Submit")
            concat_glob_button.click(concat_from_glob_horizontally, inputs=concat_glob_input, outputs=concat_glob_output)
        with gr.TabItem("Base64"):
            with gr.Row():
                base64_encode_input = gr.Textbox(lines=1, label="String")
                base64_encode_output = gr.Textbox(lines=1)
                base64_encode_button = gr.Button("Encode")
                
            with gr.Row():
                base64_decode_input = gr.Textbox(lines=1, label="Base64 String")
                base64_decode_output = gr.Textbox(lines=1)
                base64_decode_button = gr.Button("Decode")
                
            base64_encode_button.click(base64.b64encode, inputs=base64_encode_input, outputs=base64_encode_output)
            base64_decode_button.click(decode_base64_to_string, inputs=base64_decode_input, outputs=base64_decode_output)
        with gr.TabItem("Mask"):
            with gr.Row():
                mask_input = gr.Image(type="numpy")
                mask_output = gr.Image(type="numpy", interactive=False)
                log_draw_image = gr.Image(type="numpy", interactive=False) # for debugging
                gaussian_output = gr.Image(type="numpy", interactive=False) # for debugging
                debug_image_output = gr.Image(type="numpy", interactive=False) # for debugging
                pixel_input = gr.Slider(minimum=1, maximum=255, step=1, value=5, label="Pixel width")
                gaussian_input = gr.Slider(minimum=1, maximum=255, step=1, value=5, label="Gaussian blur")
                threshold_input = gr.Slider(minimum=1, maximum=255, step=1, value=40, label="Threshold")
            mask_button = gr.Button("Generate Mask")
            mask_button.click(autogen_mask_blur, inputs=[mask_input, pixel_input, threshold_input, gaussian_input], outputs=[mask_output, log_draw_image, gaussian_output])
            
            
blocks.launch()
                
                
            
        
        
