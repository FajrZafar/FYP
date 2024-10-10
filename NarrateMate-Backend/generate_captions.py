import os
import subprocess
import cv2
import surya
import json
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import sys


processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def process_image(image_path, results_dir):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Provided file is not a valid image format (png, jpg, jpeg).")

   
    subprocess.run(["surya_layout", image_path, "--images", "--results_dir", results_dir])

  
    folder_name = os.path.basename(image_path).split(".")[0]
    subfolder_path = os.path.join(results_dir, folder_name)
    json_file_path = os.path.join(subfolder_path, "results.json")
    return json_file_path


def extract_images_from_json(json_file_path, image_path, result_path, image_name):
   
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
  
    page_image = cv2.imread(image_path)
    
   
    counter = 1
    image_captions = []

    
    for entry in data.get(image_name, []):
        for bbox in entry['bboxes']:
            label = bbox['label']
            if label == 'Figure' or label == 'Picture':
                x1, y1, x2, y2 = bbox['bbox']
            
                image_region = page_image[y1:y2, x1:x2]
              
                extracted_image_path = os.path.join(result_path, f"{label}_{counter}.jpg")
                cv2.imwrite(extracted_image_path, image_region)
                
             
                image = Image.open(extracted_image_path)
                inputs = processor(images=image, return_tensors="pt")
                outputs = model.generate(**inputs)
                caption = processor.decode(outputs[0], skip_special_tokens=True)

                
                image_captions.append(f"Caption for {ordinal(counter)} image: {caption}")

                counter += 1

    return image_captions

def ordinal(n):
    suffix = ["th", "st", "nd", "rd"] + ["th"] * 16
    if 11 <= (n % 100) <= 13:
        suffix_index = 0
    else:
        suffix_index = min(n % 10, 4)
    return str(n) + suffix[suffix_index]

if __name__ == "__main__":
  
    input_path = sys.argv[1]
    results_dir = sys.argv[2]
    image_name = sys.argv[3]

   
    json_file_path = process_image(input_path, results_dir)
    
 
    image_captions = extract_images_from_json(json_file_path, input_path, results_dir, image_name)
    

    captions_file_path = os.path.join(results_dir, "captions.txt")
    with open(captions_file_path, "w", encoding="utf-8") as f:
        f.write(f"There are total {len(image_captions)} images.\n\n")
        for caption in image_captions:
            f.write(caption + "\n")
    
    print(f"Captions saved at: {captions_file_path}")
