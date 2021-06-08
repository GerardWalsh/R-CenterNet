import argparse

import os

from re import I

import sys

import time



import cv2

from imutils import paths

import torch

import numpy as np

from imutils import paths

from torch2trt import torch2trt

from torch2trt import TRTModule



sys.path.append(r"./backbone")

from resnet import ResNet



from predict import pre_process, ctdet_decode, post_process, merge_outputs

from dataset import coco_box_to_bbox

from utils.boxe import Rectangle

from utils.xml import xml_annotations_from_dict, get_lab_ret





def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--dir",

        type=str,

        help="The root folder to the streams subfolders.",

    )

    parser.add_argument(

        "--model",

        type=str,

        default="torch",

        help="The type of model for inference: tensorrt | torch",

    )

    parser.add_argument(

        "--display", type=bool, default=False, help="Whether to display detections"

    )

    parser.add_argument(

        "--optimize", type=bool, default=False, help="Whether to optimize to tensorrt"

    )

    return parser.parse_args()





def dump_bbox(rect):

    detection_ = []

    for point in rect.get_vertices_points():

        detection_.append([point.x, point.y])

    return [val for sublist in detection_ for val in sublist]





def dump_box_to_text(box, filename="tx.txt", label="defect"):

    with open(filename, "a") as outfile:

        outfile.write("%s " % label)

    for val in box:

        # print(val)

        outfile.write("%.3f " % val)

        outfile.write("\n")





def process(model, i, images, return_time=False, optimize=False):

    if optimize and i == 0:

        print("Optimizing!")

        model_ = torch2trt(model, [images], fp16_mode=True)

        torch.save(model_.state_dict(), "tensorrt_18_224_fp16.pth")

        print("Done!!!!!!!!!!!!!!!!!!!!!!")

    with torch.no_grad():

        output = model(images)

        # import ipdb



        # ipdb.set_trace()

        hm = output[0]  # .sigmoid_()

        ang = output[2]  # .relu_()

        wh = output[1]

        reg = output[3]

        forward_time = time.time()

        dets = ctdet_decode(hm, wh, ang, reg=reg, K=100)  # K



        if return_time:

            return output, dets, forward_time

        else:

            return output, dets





def iou(bbox1, bbox2, center=False):

    """Compute the iou of two boxes.

    Parameters

    ----------

    bbox1, bbox2: list.

    The bounding box coordinates: [xmin, ymin, xmax, ymax] or [xcenter, ycenter, w, h].

    center: str, default is 'False'.

    The format of coordinate.

    center=False: [xmin, ymin, xmax, ymax]

    center=True: [xcenter, ycenter, w, h]

    Returns

    -------

    iou: float.

    The iou of bbox1 and bbox2.

    """

    if center == False:

        xmin1, ymin1, xmax1, ymax1 = bbox1

        xmin2, ymin2, xmax2, ymax2 = bbox2

    else:

        xmin1, ymin1 = bbox1[0] - bbox1[2] / 2.0, bbox1[1] - bbox1[3] / 2.0

        xmax1, ymax1 = bbox1[0] + bbox1[2] / 2.0, bbox1[1] + bbox1[3] / 2.0

        xmin2, ymin2 = bbox2[0] - bbox2[2] / 2.0, bbox2[1] - bbox2[3] / 2.0

        xmax2, ymax2 = bbox2[0] + bbox2[2] / 2.0, bbox2[1] + bbox2[3] / 2.0



    # intersection

    xx1 = np.max([xmin1, xmin2])

    yy1 = np.max([ymin1, ymin2])

    xx2 = np.min([xmax1, xmax2])

    yy2 = np.min([ymax1, ymax2])



    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)

    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)



    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))

    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou





def iou_rotate_calculate(boxes1, boxes2):

    area1 = boxes1[2] * boxes1[3]

    area2 = boxes2[2] * boxes2[3]

    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])

    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])

    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

    if int_pts is not None:

        order_pts = cv2.convexHull(int_pts, returnPoints=True)

        int_area = cv2.contourArea(order_pts)

        ious = int_area * 1.0 / (area1 + area2 - int_area)

    else:

        ious = 0

    return ious





def get_pre_ret(model, i, img_path, device, conf=0.4, optimize=False):

    image = cv2.imread(img_path)

    images, meta = pre_process(image)

    images = images.to(device)

    output, dets, forward_time = process(

        model, i, images, return_time=True, optimize=optimize

    )  # output, dets, forward_time

    # import ipdb



    # ipdb.set_trace()

    dets = post_process(dets, meta)

    ret = merge_outputs(dets)



    res = np.empty([1, 7])

    for i, c in ret.items():

        tmp_s = ret[i][ret[i][:, 5] > conf]

        tmp_c = np.ones(len(tmp_s)) * (i + 1)

        tmp = np.c_[tmp_c, tmp_s]

        res = np.append(res, tmp, axis=0)

        res = np.delete(res, 0, 0)

        res = res.tolist()

    # ipdb.set_trace()

    return res, image





def pre_recall(model, root_path, device, display=False, iou=0.5, optimize=False):

    imgs = paths.list_images(root_path)

    num = 0

    all_pre_num = 0

    all_lab_num = 0

    miou = 0

    mang = 0

    ll = [x for x in imgs]

    ll.sort()



    flip = False



    for i, img in enumerate(ll):

        img = img.split("/")[-1]

        if img.split(".")[-1] == "jpg":

            detection_lol = []

            label_lol = []

            img_path = os.path.join(root_path, img)

            xml_path = os.path.join(root_path, img.split(".")[0] + ".xml")

            detections, image = get_pre_ret(

                model, i, img_path, device, optimize=optimize

            )

            if flip:

                image = cv2.flip(image, 1)

            if display:

                for v, detect in enumerate(detections):

                    class_name, lx, ly, rx, ry, ang, prob = detect

                    detection_list = [

                        (rx + lx) / 2,

                        (ry + ly) / 2,

                        (rx - lx),

                        (ry - ly),

                        ang,

                    ]

                    detection_lol.append(detection_list)

                    # detection = np.array(detection_list)

                    # detection_rect = Rectangle(*detection, image, colour=(0, 255, 255))

                    # hold = img_path.split('.')[0].split('/')

                    # detection_txt_file = "/".join(hold[0:5]) + '/detections/' + hold[-1] + '.txt'



                for box in detection_lol:

                    cv2.circle(

                        image, (int(box[0]), int(box[1])), 2, (255, 0, 0), -1

                    )  # blue

                    # cv2.circle(image, (int(box[]), int(box[1])), 2, (0, 0, 255), -1) # red

                    detection_rect = Rectangle(*box, image, flip, colour=(0, 255, 255))

                    detection_rect.draw(image, flip)



                # splits = img_path.split('/')

                # print(splits[-1])

                # new_image_path = splits[-1]

                # new_label_path = new_image_path.split('.')[0] + '.xml'

                cv2.imshow("test", image)

                key = cv2.waitKey(1)

                if key == ord("q"):

                    break



    # return miou/num





if __name__ == "__main__":

    import time



    args = parse_args()

    print(f"Model specified: {args.model}")

    if args.model == "torch":

        # Pytorch model

        model = ResNet(18)

        model.load_state_dict(torch.load("./last_18_224.pth"))

    elif args.model == "tensorrt":

        # TensorRT model

        model = TRTModule()

        model.load_state_dict(torch.load("tensorrt_18_224_fp16.pth"))

    else:

        raise "Model type not defined"



    # Common model setup ops

    device = torch.device("cuda")

    model.eval()

    model.cuda()



    start = time.time()

    miou = pre_recall(model, args.dir, device, args.display, optimize=args.optimize)

    frames = len([path for path in paths.list_images(args.dir)])

    end = time.time()

    print("FPS:{}".format(frames / (end - start)))


