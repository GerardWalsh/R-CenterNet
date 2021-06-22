import argparse
import time
import math
import time
import os
import sys
import pathlib

import cv2
import torch
from torch2trt import torch2trt
import evaluation
import numpy as np
from imutils import paths
import copy

sys.path.append(r"./backbone")

# from dlanet_dcn import DlaNet
# from resnet_dcn import ResNet

from resnet import ResNet

# from predict import pre_process, ctdet_decode, post_process, merge_outputs
# from dataset import coco_box_to_bbox
# from utils.boxes import Rectangle
# from utils.xml import xml_annotations_from_dict, get_lab_ret


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--dir",
    #     type=str,
    #     required=True,
    #     help="The path to the test images.",
    # )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to model to perform inference with.",
    )
    # parser.add_argument(
    #     "--confidence-threshold",
    #     type=float,
    #     default=0.5,
    #     help="Threshold upon which to discard detections.",
    # )
    # parser.add_argument(
    #     "--input-size",
    #     type=int,
    #     default=512,
    #     help="Size of image model was trained on.",
    # )
    parser.add_argument(
        "--model-arch",
        type=str,
        default="resnet_18",
        help="Model architecture type.",
    )
    # parser.add_argument(
    #     "--save-predictions",
    #     type=bool,
    #     default=False,
    #     help="Whether to store model predictions.",
    # )
    # parser.add_argument(
    #     "--create-gt",
    #     type=bool,
    #     default=False,
    #     help="Whether to create gt labels.",
    # )
    # parser.add_argument(
    #     "--visualize",
    #     type=bool,
    #     default=False,
    #     help="Whether to create gt labels.",
    # )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("INFO: Creating model . . . . .")
    model = ResNet(int(args.model_arch.split("_")[-1]))
    device = torch.device("cuda")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.cuda()
    print("INFO: Model initialised and in GPU memory . . . . .")

    print("INFO: Creating dummy input . . . . .")
    x = torch.ones((1, 3, 512, 512)).cuda()
    print("INFO: Optimising model . . . . .")
    model_trt = torch2trt(model, [x])
    print("INFO: Saving optimised model . . . . .")
    torch.save(model_trt.state_dict(), "centernet_optimised.pth")

    testing_runs = 20
    print("INFO: Testing standard inference . . . . .")
    start_time = time.time()
    for i in range(testing_runs):
        _ = model(x)
    end_time = time.time()
    print(f"FPS: {(end_time - start_time) / testing_runs}")

    print("INFO: Testing optimised inference . . . . .")
    start_time = time.time()
    for i in range(testing_runs):
        _ = model_trt(x)
    end_time = time.time()
    print(f"FPS: {(end_time - start_time) / testing_runs}")

    # miou = pre_recall(
    #     args.dir,
    #     device,
    #     input_size=args.input_size,
    #     store_predictions=args.save_predictions,
    #     create_gt=args.create_gt,
    #     visualize=args.visualize,
    # )
    # pre_recall('../tests/t3', device)
    # print('Mean average IOU:', miou)
    # pre_recall('./test_frames', device)
    # F1 = (2*p*r)/(p+r)
    # print(F1)
