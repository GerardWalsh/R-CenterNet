import os

label_dir = '/home/gexegetic/R-CenterNet/data/defect/annotations/'
image_dir = '/home/gexegetic/R-CenterNet/data/images/'
new_dir = '/home/gexegetic/R-CenterNet/data/defect/images/'

for count, filename in enumerate(os.listdir(label_dir)): 
    # print(filename)
    splits = filename.split('.')
    print(splits[0])
    image_filename = splits[0] + '.jpg'
    print(image_dir + image_filename)

    # file_split = splits[-1].split('.')

    # new_filename = str(int(file_split[0])+3000) + '.' + file_split[-1]
    # print(new_filename)
    os.rename(image_dir + image_filename, new_dir + image_filename) 