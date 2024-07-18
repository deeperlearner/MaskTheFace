import random
import os
import argparse
import io
import struct

import dlib
import mxnet as mx
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.aux_functions import *

# Command-line input setup
parser = argparse.ArgumentParser(
    description="MaskTheFace - Python code to mask faces dataset"
)
parser.add_argument(
    "--path",
    type=str,
    default="",
    help="Path to either the folder containing images or the image itself",
)
parser.add_argument(
    "--mask_type",
    type=str,
    default="surgical",
    choices=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"],
    help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
)

parser.add_argument(
    "--pattern",
    type=str,
    default="",
    help="Type of the pattern. Available options in masks/textures",
)

parser.add_argument(
    "--pattern_weight",
    type=float,
    default=0.5,
    help="Weight of the pattern. Must be between 0 and 1",
)

parser.add_argument(
    "--color",
    type=str,
    default="#0473e2",
    help="Hex color value that need to be overlayed to the mask",
)

parser.add_argument(
    "--color_weight",
    type=float,
    default=0.5,
    help="Weight of the color intensity. Must be between 0 and 1",
)

parser.add_argument(
    "--code",
    type=str,
    default="cloth-#ffc0cb, cloth-#000000, cloth-#ffffff, cloth, cloth-masks/textures/others/heart_1.png, KN95, surgical_blue, surgical_green, surgical-#ffffff, surgical-#000000",
    # default="",
    help="Generate specific formats",
)


parser.add_argument(
    "--verbose", dest="verbose", action="store_true", help="Turn verbosity on"
)
parser.add_argument(
    "--write_original_image",
    dest="write_original_image",
    action="store_true",
    help="If true, original image is also stored in the masked folder",
)
parser.set_defaults(feature=False)

args = parser.parse_args()
args.write_path = args.path#+ "_masked"

# Set up dlib face detector and predictor
args.detector = dlib.get_frontal_face_detector()
path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(path_to_dlib_model):
    download_dlib_model()

args.predictor = dlib.shape_predictor(path_to_dlib_model)

# Extract data from code
mask_code = "".join(args.code.split()).split(",")
args.code_count = np.zeros(len(mask_code))
args.mask_dict_of_dict = {}


for i, entry in enumerate(mask_code):
    mask_dict = {}
    mask_color = ""
    mask_texture = ""
    mask_type = entry.split("-")[0]
    if len(entry.split("-")) == 2:
        mask_variation = entry.split("-")[1]
        if "#" in mask_variation:
            mask_color = mask_variation
        else:
            mask_texture = mask_variation
    mask_dict["type"] = mask_type
    mask_dict["color"] = mask_color
    mask_dict["texture"] = mask_texture
    args.mask_dict_of_dict[i] = mask_dict

# Path to the RecordIO files
input_index_recordio_path = '/media/back/internal/data/ms1m-retinaface-t1/train.idx'
input_recordio_path = '/media/back/internal/data/ms1m-retinaface-t1/train.rec'

output_index_recordio_path = '/space/data/ms1m-retinaface-t1_v1_all_mask/train.idx'
output_recordio_path = '/space/data/ms1m-retinaface-t1_v1_all_mask/train.rec'
mask_index_recordio_path = '/space/data/ms1m-retinaface-t1_v1_all_mask/mask.idx'
mask_recordio_path = '/space/data/ms1m-retinaface-t1_v1_all_mask/mask.rec'

total_records = 5179510

# Read the input RecordIO file
record = mx.recordio.MXIndexedRecordIO(input_index_recordio_path, input_recordio_path, 'r')
# Open the output RecordIO file
record_out = mx.recordio.MXIndexedRecordIO(output_index_recordio_path, output_recordio_path, 'w')
record_mask = mx.recordio.MXIndexedRecordIO(mask_index_recordio_path, mask_recordio_path, 'w')

# Read and process each record
for _ in tqdm(range(total_records), desc="Writing records"):
    idx = _ + 1
    # if idx < 5179500:
    #     continue
    item = record.read_idx(idx)
    if item is None:
        break  # End of file
    header, image_data = mx.recordio.unpack(item)

    try:
        mode = "normal"
        image = mx.image.imdecode(image_data).asnumpy()
    except:
        mode = "abnormal"
        print(mode)
        break

    # Modify the image (e.g., resize, rotate, etc.)
    image_path = ""
    masked_image, mask, mask_binary_array, original_image = mask_image(
        image_path, args, array_img=image
    )
    if len(masked_image):
        masked_image = Image.fromarray(np.uint8(masked_image[0]))
        mask_binary_array = mask_binary_array[0]
    else:
        continue

    # Convert the modified image back to bytes
    buffer = io.BytesIO()
    masked_image.save(buffer, format='JPEG')
    modified_image_data = buffer.getvalue()

    # Write the modified image back to the RecordIO file
    packed_record = mx.recordio.pack(header, modified_image_data)
    record_out.write_idx(idx, packed_record)

    # Convert the additional array to a grayscale image
    grayscale_image = Image.fromarray(mask_binary_array, mode='L')
    buffer_gray = io.BytesIO()
    grayscale_image.save(buffer_gray, format='JPEG')
    grayscale_image_data = buffer_gray.getvalue()

    # Write the modified image back to the RecordIO file
    packed_record = mx.recordio.pack(header, grayscale_image_data)
    record_mask.write_idx(idx, packed_record)

record.close()
record_out.close()
record_mask.close()


# Read the output RecordIO file
record_out = mx.recordio.MXIndexedRecordIO(output_index_recordio_path, output_recordio_path, 'r')
record_mask = mx.recordio.MXIndexedRecordIO(mask_index_recordio_path, mask_recordio_path, 'r')

# Read and process each record
for _ in tqdm(range(total_records), desc="Reading records"):
    idx = _ + 1
    # if idx < 5179500:
    #     continue
    if idx > 50:
        break
    item = record_out.read_idx(idx)
    if item is None:
        break  # End of file
    header, image_data = mx.recordio.unpack(item)
    print(header)

    # Convert image data to a PIL image
    try:
        image = mx.image.imdecode(image_data).asnumpy()
        image = Image.fromarray(image)
        image.save(f"tests/image_{idx}.jpg")
    except Exception as e:
        print(f"Error opening image: {e}")
        continue

    item = record_mask.read_idx(idx)
    if item is None:
        break  # End of file
    header, mask_data = mx.recordio.unpack(item)

    # Convert grayscale image data to a PIL image
    try:
        grayscale_image = mx.image.imdecode(mask_data, flag=0).asnumpy()
        grayscale_image = grayscale_image.reshape((112, 112))
        grayscale_image = Image.fromarray(grayscale_image, mode='L')
        grayscale_image.save(f"tests/gray_{idx}.jpg")
    except Exception as e:
        print(f"Error opening grayscale image: {e}")

record_out.close()
record_mask.close()
