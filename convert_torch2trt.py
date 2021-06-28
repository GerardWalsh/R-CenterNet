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
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to model to perform inference with.",
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
        "--head-conv",
        type=int,
        default=256,
        help="Model architecture type.",
    )
    parser.add_argument(
        "--int8-mode",
        type=bool,
        default=False,
        help="Whether to use int8 precision.",
    )
    parser.add_argument(
        "--fp16-mode",
        type=bool,
        default=False,
        help="Whether to use half precision.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("INFO: Creating model . . . . .")
    model = ResNet(int(args.model_arch.split("_")[-1]), head_conv=args.head_conv)
    device = torch.device("cuda")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.cuda()
    print("INFO: Model initialised and in GPU memory . . . . .")

    print("INFO: Creating dummy input . . . . .")
    x = torch.ones((1, 3, args.input_size, args.input_size)).cuda()
    print("INFO: Optimising model . . . . .")
    model_trt = torch2trt(model, [x], int8_mode=args.int8_mode, fp16_mode=args.fp16_mode)
    print("INFO: Saving optimised model . . . . .")
    torch.save(model_trt.state_dict(), "centernet_optimised.pth")

    testing_runs = 20
    print("INFO: Testing standard inference . . . . .")
    start_time = time.time()
    for i in range(testing_runs):
        _ = model(x)
    end_time = time.time()
    print(f"FPS: {testing_runs / (end_time - start_time)}")

    print("INFO: Testing optimised inference . . . . .")
    start_time = time.time()
    for i in range(testing_runs):
        _ = model_trt(x)
    end_time = time.time()
    print(f"FPS: {testing_runs / (end_time - start_time)}")
