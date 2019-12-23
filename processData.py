'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script processes the COCO dataset
'''  

import os
import pickle
from PIL import Image
from tqdm import tqdm

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def main():
    cap2image_file = "data/cap2image.pickle"
    cap2ids = pickle.load(open(cap2image_file, "rb"))
    id2path_file = "data/id2path.pickle"
    id2path = pickle.load(open(id2path_file, "rb"))
    all_ids = []
    for _, ids in cap2ids.items():
        all_ids.extend(ids)
    all_ids = set(all_ids)
    all_path = [id2path[id] for id in all_ids if id != 0]
    folder = 'flickr30k-images'
    resized_folder = 'flickr30k-images-resized-select-top5'
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    image_files = os.listdir(folder)
    for i, image_file in enumerate(tqdm(image_files, desc="image")):
        if image_file in all_path:
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image.save(os.path.join(resized_folder, image_file), image.format)

    print("done resizing images...")
main()