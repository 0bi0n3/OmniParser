import os
import torch
import json
import base64
import io
from PIL import Image
from tqdm import tqdm # A library for creating smart progress bars

# --- Make sure your utility functions are importable ---
from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

# ==============================================================================
# 1. SETUP - This is the same setup code you had before
# ==============================================================================
print("Setting up models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Detection Model (YOLO)
detection_model_path = 'weights/icon_detect/model.pt'
som_model = get_yolo_model(detection_model_path)
som_model.to(device)

# Load Caption Model (Florence-2)
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence",
    device=device
)
print(f"Models loaded successfully on '{device}'.")

# ==============================================================================
# 2. CONFIGURATION - Define your input/output folders and settings
# ==============================================================================
INPUT_FRAMES_DIR = "input_frames"         # Folder containing your extracted video frames
OUTPUT_IMAGE_DIR = "outputs/images" # Folder to save annotated images
OUTPUT_JSON_DIR = "outputs/json"    # Folder to save the structured data

# Create output directories if they don't exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# Model inference settings
BOX_TRESHOLD = 0.05
# Note: box_overlay_ratio is now calculated inside the loop for each image

# ==============================================================================
# 3. BATCH PROCESSING - The main loop
# ==============================================================================
print("Starting batch processing...")
# Get a list of all image files in the input directory
image_files = [f for f in os.listdir(INPUT_FRAMES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each file with a tqdm progress bar
for filename in tqdm(image_files, desc="Processing Frames"):
    try:
        base_filename = os.path.splitext(filename)[0]
        input_image_path = os.path.join(INPUT_FRAMES_DIR, filename)
        
        # Define where the output files for this frame will go
        output_image_path = os.path.join(OUTPUT_IMAGE_DIR, f"{base_filename}_annotated.jpg")
        output_json_path = os.path.join(OUTPUT_JSON_DIR, f"{base_filename}_data.json")

        # --- RESUME LOGIC ---
        # If the JSON file already exists, skip this frame
        if os.path.exists(output_json_path):
            continue

        # --- DYNAMIC BBOX CONFIG ---
        # Open the image to calculate the dynamic overlay ratio
        with Image.open(input_image_path) as temp_img:
            box_overlay_ratio = max(temp_img.size) / 3200
        
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # --- OCR PASS ---
        ocr_bbox_rslt, _ = check_ocr_box(
            input_image_path,
            display_img=False,
            output_bb_format='xyxy',
            use_paddleocr=True,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        text, ocr_bbox = ocr_bbox_rslt if ocr_bbox_rslt is not None else ([], [])

        # --- MAIN PARSING ---
        # Ensure the function returns a value before unpacking
        result = get_som_labeled_img(
            input_image_path,
            som_model,
            BOX_TRESHOLD=BOX_TRESHOLD,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128
        )
        
        if result is None:
            print(f"Skipping {filename}: No elements found.")
            continue
            
        annotated_image, label_coordinates, parsed_content_list = result

        # --- SAVE THE RESULTS ---
        # Move the annotated image from its temporary location
        image = Image.open(io.BytesIO(base64.b64decode(annotated_image)))

        with open(output_image_path, 'w') as f:
            image.save(output_image_path)

        # Save the structured data to a JSON file
        with open(output_json_path, 'w') as f:
            json.dump(parsed_content_list, f, indent=4)

    except Exception as e:
        print(f"Failed to process {filename}. Error: {e}")
        # Continue to the next file even if one fails
        continue

print("Batch processing complete!")
