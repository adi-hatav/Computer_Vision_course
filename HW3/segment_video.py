import os
import glob
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


# This file segments frames in a folder using the SAM model with a point + bounding box prompt


ALL_FRAMES = True # processing all the final frames chosen for the video
NORMALIZE = False # normalize frame dimensions to height 250 while keeping aspect ratio
SAVEDIR = 'HW3/HW3 final/data/my_data/juggling_segmented_hd2'
print("ALL_FRAMES:", ALL_FRAMES)
print("NORMALIZE:", NORMALIZE)


init_time = time.time()


# Load SAM model
print("Loading SAM model...")
start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
checkpoint = "HW3/sam_vit_h_4b8939.pth"
model_type = "vit_h"

torch.cuda.empty_cache()

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
print(f"Model loaded in {time.time() - start:.2f} seconds")


# Load video frames
print("Loading video frames...")
start = time.time()
juggling_frames = []
for img in os.listdir('HW3/HW3 final/data/my_data/juggling_frames'):
    if 165 < int(img[:4]) <= 405:
        juggling_frames.append(cv2.imread(f'HW3/HW3 final/data/my_data/juggling_frames/{img}'))

# resize frames
if NORMALIZE:
    juggling_frames = [cv2.resize(frame, (int(frame.shape[1] / frame.shape[0] * 250), 250)) for frame in juggling_frames]

print(f"Number of frames: {len(juggling_frames)}")
print(f"Frames loaded in {time.time() - start:.2f} seconds")


# Segment video
print("Segmenting video...")

if NORMALIZE:
    # normalized frame parameters
    point_human = np.array([[73, 173]])
    input_label = np.array([1])
    box_human = np.array([45, 111, 105, 250])
else:
    # large frame parameters
    point_human = np.array([[244, 585], [313, 820], [258, 425], [328, 528]])
    input_label = np.array([1, 1, 1, 1])
    box_human = np.array([148, 364, 390, 832]) # originally 162, 364, 390, 832

for ind, frame in enumerate(juggling_frames):
    # change this criterion to process a subset of the frames
    # if 132 < ind < 136 or 138 < ind < 144:
    print(f"Processing frame {ind}...")
    start = time.time()
    predictor.set_image(frame)
    print("Finished encoding image")

    mask, scores, logits = predictor.predict(
        point_coords=point_human,
        point_labels=input_label,
        box=box_human,
        multimask_output=False
    )

    mask_uint = (mask.squeeze() * 255).astype(np.uint8)
    
    # Add alpha channel to original frame
    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Scale mask to match frame sizing
    frame_height, frame_width = frame_bgra.shape[0:2]
    resized_mask = cv2.resize(mask_uint, dsize = (frame_width, frame_height))

    # Use the mask as the alpha channel of the frame
    frame_bgra[:,:,3] = resized_mask

    # frame[mask.squeeze() != 1] = [255, 255, 255] # white
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # create the directory if it does not exist
    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)

    cv2.imwrite(f'{SAVEDIR}/{ind:04d}.png', frame_bgra)
    print(f"Processed frame {ind} in {time.time() - start:.2f} seconds")

    torch.cuda.empty_cache()


print(f"Segmentation finished in {time.time() - init_time:.2f} seconds")

