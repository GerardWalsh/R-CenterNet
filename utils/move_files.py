import os

label_dir = "/home/gerwal/train/R-CenterNet/data/defect/annotations_val_small_set"
image_dir = "/home/gerwal/train/R-CenterNet/data/images_small_set/"
new_dir = "/home/gerwal/train/R-CenterNet/data/d/"

for count, filename in enumerate(os.listdir(label_dir)):
    splits = filename.split(".")
    image_filename = splits[0] + ".jpg"
    try:
        os.rename(image_dir + image_filename, new_dir + image_filename)
    except:
        print("file not found")
