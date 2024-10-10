

#final code for text extraction- sending it to frontend
import torch
import surya
import json
from collections import defaultdict
from surya.input.langs import replace_lang_with_code, get_unique_langs
from surya.input.load import load_from_file
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.tokenizer import _tokenize
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
import os
from gtts import gTTS

def extract_text(input_path, results_dir, langs='hi,en', images=False):
   
    torch.mps.empty_cache()
    
    print(f"Received input path: {input_path}")
    print(f"Results directory: {results_dir}")

    images, names = load_from_file(input_path)

    folder_name = os.path.basename(input_path).split(".")[0]
    result_path = os.path.join(results_dir, folder_name)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    os.makedirs(result_path, exist_ok=True)

    langs = langs.split(",")
    replace_lang_with_code(langs)
    image_langs = [langs] * len(images)

    det_processor = load_detection_processor()
    det_model = load_detection_model()

    _, lang_tokens = _tokenize("", get_unique_langs(image_langs))
    rec_model = load_recognition_model(langs=lang_tokens)
    rec_processor = load_recognition_processor()

    predictions_by_image = run_ocr(images, image_langs, det_model, det_processor, rec_model, rec_processor)

    extracted_text = ""
    if images:
        for idx, (name, _, pred, _) in enumerate(zip(names, images, predictions_by_image, image_langs)):
    
            filtered_text_lines = [line.text for line in pred.text_lines if "Click https://bit.ly/FG-Books to download all PDF FG books for FREE" not in line.text]
            extracted_text += ' '.join(filtered_text_lines) + "\n"

    
        with open(os.path.join(result_path, "extracted_text.txt"), "w", encoding="utf-8") as f:
            f.write(extracted_text)

    return result_path, extracted_text

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]
    results_dir = sys.argv[2]
    result_path, extracted_text = extract_text(input_path, results_dir)

