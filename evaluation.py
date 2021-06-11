import argparse
import math
import time
import os
import sys

import cv2
import torch
import evaluation
import numpy as np
from imutils import paths
import copy

sys.path.append(r"./backbone")

# from dlanet_dcn import DlaNet
# from resnet_dcn import ResNet

from resnet import ResNet

from predict import pre_process, ctdet_decode, post_process, merge_outputs
from dataset import coco_box_to_bbox
from utils.boxes import Rectangle
from utils.xml import xml_annotations_from_dict, get_lab_ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="The path to the test images.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to model to perform inference with.",
    )
    # parser.add_argument(
    #     "--display", type=bool, default=False, help="Whether to display detections"
    # )
    # parser.add_argument(
    #     "--optimize", type=bool, default=False, help="Whether to optimize to tensorrt"
    # )
    return parser.parse_args()


# from rioy import rbox_iou


def flip_data(img, bboxes):
    # bboxes = [bbox, bbox]
    img_center = np.array(img.shape[:2])[::-1] / 2
    img_center = np.hstack((img_center, img_center))
    # if random.random() < self.p:
    # img =  img[:,::-1,:]

    bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])
    box_w = abs(bboxes[:, 0] - bboxes[:, 2])

    bboxes[:, 0] -= box_w
    bboxes[:, 2] += box_w

    # print(bboxes)

    # bboxes[:, -1] *= -1

    return cv2.flip(img, 1), bboxes


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


def process(images, return_time=False):
    with torch.no_grad():
        output = model(images)
        hm = output["hm"].sigmoid_()
        #   print('Heatmap shape', hm.shape)
        #   print('Heatmap ', hm)
        ang = output["ang"]  # .relu_()
        #   print("angle shape in process", ang.shape)
        wh = output["wh"]
        #   print("WH shape in process", wh.shape)
        reg = output["reg"]
        #   print("REG shape in process", reg.shape)

        #   torch.cuda.synchronize()
        forward_time = time.time()
        dets = ctdet_decode(hm, wh, ang, reg=reg, K=100)  # K
    #   print('Heatmap shape', hm.shape)

    if return_time:
        return output, dets, forward_time
    else:
        return output, dets


def get_pre_ret(img_path, device, conf=0.5):
    image = cv2.imread(img_path)
    # image = cv2.resize(image, (960, 540))
    images, meta = pre_process(image, image_size=512)
    images = images.to(device)
    output, dets, forward_time = process(images, return_time=True)

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
    return res, image


def pre_recall(root_path, device, iou=0.5):
    imgs = paths.list_images(root_path)
    num = 0
    all_pre_num = 0
    all_lab_num = 0
    miou = 0
    mang = 0
    ll = [x for x in imgs]
    ll.sort()
    # print("images", ll)

    flip = False

    for i, img in enumerate(ll):
        img = img.split("/")[-1]
        # print('IMG', img)
        if img.split(".")[-1] == "jpg":
            detection_lol = []
            label_lol = []
            img_path = os.path.join(root_path, img)
            # print('Image filepath', img_path)
            xml_path = os.path.join(root_path, img.split(".")[0] + ".xml")
            detections, image = get_pre_ret(img_path, device)
            if flip:
                image = cv2.flip(image, 1)

            # labels = get_lab_ret(xml_path)
            # all_pre_num += len(detections)
            # all_lab_num += len(labels)

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
                detection = np.array(detection_list)
                detection_rect = Rectangle(*detection, image, colour=(0, 255, 255))
                hold = img_path.split(".")[0].split("/")
                detection_txt_file = (
                    "/".join(hold[0:5]) + "/detections/" + hold[-1] + ".txt"
                )
                # print('Detection text file', detection_txt_file)
                # dump_box_to_text([prob] + detection_list, detection_txt_file)

                # for point in detection_rect.get_vertices_points():
                #     # print(point.x, point.y)
                #     if ang > 0:
                #         cv2.circle(image, (point.x, point.y), 2, (255, 0, 0), -1)
                #     else:
                #         cv2.circle(image, (point.x, point.y), 2, (0, 255, 0), -1)

            # print('im shape', image.shape)

            # for cx, cy, w, h, ang_l in labels:
            #     # if flip:
            #     #     cx = image.shape[1] - cx
            #     if ang_l > math.pi:
            #         ang_l = -(2. * math.pi - ang_l)
            #         # ang_l = -1* ang_l
            #     label = [cx, cy, w, h, ang_l]
            #     # label_lol.append(label)
            #     # label_txt_file = "/".join(hold[0:5]) + '/gt/' + hold[-1] + '.txt'
            #     # print('Detection text file', label_txt_file)
            #     # dump_box_to_text(label, label_txt_file)

            #     label_rect = Rectangle(*label, image, flip=flip, colour=(0, 255, 0))
            #     label_rect.draw(image)
            # for point in label_rect.get_vertices_points():
            #     # print(point.x, point.y)
            #     if ang_l > 0:
            #         cv2.circle(image, (point.x, point.y), 2, (255, 0, 0), -1)
            #     else:
            #         cv2.circle(image, (point.x, point.y), 2, (0, 255, 0), -1)

            #         cv2.imshow('test {}'.format(v), demo_img)
            #         key = cv2.waitKey()
            #         if key == ord('q'):
            #             break
            #         elif key == ord('m'):
            #             print(rbox_iou(detection_list, label))
            #             cv2.destroyAllWindows()
            #             break

            # cv2.destroyAllWindows()

            # iou = iou_rotate_calculate(detection, lab_one)
            # detection_lol = np.vstack(detection_lol[0:1])
            # # print(detection_lol)
            # # image, boxes = flip_data(image, detection_lol)
            # if flip:
            #     flip_image = cv2.flip(image, 1)

            for box in detection_lol:
                cv2.circle(
                    image, (int(box[0]), int(box[1])), 2, (255, 0, 0), -1
                )  # blue
                # cv2.circle(image, (int(box[]), int(box[1])), 2, (0, 0, 255), -1) # red
                detection_rect = Rectangle(*box, image, flip, colour=(0, 255, 255))
                detection_rect.draw(image, flip)

            # cv2.imshow('flipped', flip_image)

            # for box in detection_lol:
            #     cv2.circle(image, (int(box[0]), int(box[1])), 2, (255, 0, 0), -1) # blue
            #     # cv2.circle(image, (int(box[]), int(box[1])), 2, (0, 0, 255), -1) # red
            #     detection_rect = Rectangle(*box, image, colour=(0, 255, 255))
            #     detection_rect.draw(image, flip=False)
            # path_split = img_path.split('/')

            # # print('image path', img_path)
            splits = img_path.split("/")
            # print(splits[-1])
            new_image_path = splits[-1]
            new_label_path = new_image_path.split(".")[0] + ".xml"
            # print(new_label_path)
            # xml_annotations_from_dict(
            #     input_dict={"rotated_boxes": detection_lol, "classes": ["defect"]*len(detection_lol)},
            #     output_dir="/home/gexegetic/R-CenterNet/VID_20210119_123453/",
            #     image_fpath=new_image_path,
            #     image_shape=image.shape,
            #     image_fname=splits[-1]
            #     )

            # cv2.imshow('original', image)
            # print('Saving image to', "/home/gexegetic/R-CenterNet/flipped/" + new_image_path)
            # cv2.imwrite("/home/gexegetic/R-CenterNet/flip/" + new_image_path, image)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break
            # print("IOU", iou)
            # ang_err = abs(ang - ang_l)/180
            # if iou > 0.5:
            #     num += 1
            #     miou += iou
            #     mang += ang_err

        # print('Detection lol', detection_lol)
        # cv2.imwrite('./convince/frame_' + str(i).zfill(10) + '.jpg', image)
        cv2.imshow("test", image)
        key = cv2.waitKey()
        if key == ord("q"):
            break

    # return miou/num


if __name__ == "__main__":
    args = parse_args()
    model = ResNet(34)
    # model = DlaNet(34)
    device = torch.device("cuda")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.cuda()

    miou = pre_recall(args.dir, device)
    # pre_recall('../tests/t3', device)
    # print('Mean average IOU:', miou)
    # pre_recall('./test_frames', device)
    # F1 = (2*p*r)/(p+r)
    # print(F1)
