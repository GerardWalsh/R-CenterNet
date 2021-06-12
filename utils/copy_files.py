import os
from shutil import copyfile

label_dir = "./data/defect/annotations_val_small_set/"
image_dir = "./data/defect/images_small_set/"
new_dir = "./data/defect/validation_images/"

for count, filename in enumerate(os.listdir(label_dir)):
    splits = filename.split(".")
    image_filename = splits[0] + ".jpg"
    copyfile(image_dir + image_filename, new_dir + image_filename)
