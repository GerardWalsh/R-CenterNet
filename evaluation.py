import argparse
import math
import time
import os
import sys
import pathlib

import cv2
import torch
from torch2trt import TRTModule
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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Threshold upon which to discard detections.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=512,
        help="Size of image model was trained on.",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default="resnet_18",
        help="Model architecture type.",
    )
    parser.add_argument(
        "--save-predictions",
        type=bool,
        default=False,
        help="Whether to store model predictions.",
    )
    parser.add_argument(
        "--create-gt",
        type=bool,
        default=False,
        help="Whether to create gt labels.",
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="Whether to create gt labels.",
    )
    parser.add_argument(
        "--head-conv",
        type=int,
        default=256,
        help="Conv filters in regression heads.",
    )
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


def process(model, images, return_time=False):
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


def get_pre_ret(model, img_path, device, conf=0.3, input_size=224):
    image = cv2.imread(img_path)
    # image = cv2.resize(image, (960, 540))
    images, meta = pre_process(image, image_size=input_size)
    images = images.to(device)
    output, dets, forward_time = process(model, images, return_time=True)

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


def pre_recall(
    model,
    root_path,
    device,
    input_size,
    iou=0.5,
    store_predictions=False,
    create_gt=False,
    visualize=False,
):
    imgs = paths.list_images(root_path)
    ll = [x for x in imgs]
    ll.sort()
    print(f"Got {len(ll)} images")

    flip = False
    # Create directory for predictions
    if store_predictions:
        prediction_dir = pathlib.Path(root_path) / "predictions"
        prediction_dir.mkdir()
        print(f"Created {str(prediction_dir)} for saving model output.")
    if create_gt:
        gt_dir = pathlib.Path(root_path) / "gt"
        gt_dir.mkdir()
        print(f"Created {str(gt_dir)} for saving model output.")

    for i, img in enumerate(ll):
        img = img.split("/")[-1]
        if img.split(".")[-1] == "jpg":
            detection_lol = []
            label_lol = []
            img_path = os.path.join(root_path, img)
            # print('Image filepath', img_path)
            xml_path = os.path.join(root_path, img.split(".")[0] + ".xml")
            detections, image = get_pre_ret(
                model, img_path, device, input_size=input_size
            )
            if flip:
                image = cv2.flip(image, 1)

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
                hold = img_path.split(".")[0].split("/")
                if store_predictions:
                    print("Saving predictions")
                    # detection_rect = Rectangle(*detection, image, colour=(0, 255, 255))
                    detection_txt_file = prediction_dir / (hold[-1] + ".txt")
                    # print("Detection text file", detection_txt_file)
                    dump_box_to_text([prob] + detection_list, detection_txt_file)

            if create_gt:
                print("Saving gt")
                labels = get_lab_ret(xml_path)
                for cx, cy, w, h, ang_l in labels:
                    # if flip:
                    #     cx = image.shape[1] - cx
                    label = [cx, cy, w, h, ang_l]
                    # label_lol.append(label)
                    label_txt_file = gt_dir / (hold[-1] + ".txt")
                    # print('Detection text file', label_txt_file)
                    dump_box_to_text(label, label_txt_file)

                    label_rect = Rectangle(*label, image, flip=flip, colour=(0, 255, 0))
                    label_rect.draw(image)

            if args.visualize:
                for box in detection_lol:
                    cv2.circle(
                        image, (int(box[0]), int(box[1])), 2, (255, 0, 0), -1
                    )  # blue
                    # cv2.circle(image, (int(box[]), int(box[1])), 2, (0, 0, 255), -1) # red
                    detection_rect = Rectangle(*box, image, colour=(0, 255, 255))
                    detection_rect.draw(image, flip=False)

        if args.visualize:
            cv2.imshow("test", image)
            key = cv2.waitKey()
            if key == ord("q"):
                break


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda")

    # model = ResNet(int(args.model_arch.split("_")[-1]), head_conv=args.head_conv)
    # model.load_state_dict(torch.load(args.model_path))
    # model.eval()

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(args.model_path))
    model_trt.cuda()

    miou = pre_recall(
        model_trt,
        args.dir,
        device,
        input_size=args.input_size,
        store_predictions=args.save_predictions,
        create_gt=args.create_gt,
        visualize=args.visualize,
    )
    # pre_recall('../tests/t3', device)
    # print('Mean average IOU:', miou)
    # pre_recall('./test_frames', device)
    # F1 = (2*p*r)/(p+r)
    # print(F1)
