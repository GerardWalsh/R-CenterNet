import os

label_dir = "/home/gexegetic/R-CenterNet/data/defect/annotations/"
image_dir = "/home/gexegetic/R-CenterNet/data/images/"
new_dir = "/home/gexegetic/R-CenterNet/data/defect/images/"

for count, filename in enumerate(os.listdir(label_dir)):
    splits = filename.split(".")
    image_filename = splits[0] + ".jpg"
    os.rename(image_dir + image_filename, new_dir + image_filename)
