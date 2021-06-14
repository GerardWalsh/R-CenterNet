import argparse
import sys
import re

import torch.onnx

sys.path.append(r"./backbone")

from resnet import ResNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the model to convert from torch to onnx.",
    )
    parser.add_argument(
        "--model-arch",
        type=int,
        required=True,
        help="The resnet model architecture.",
    )
    parser.add_argument(
        "--model-input-size",
        type=int,
        required=True,
        help="The input size used to train the model.",
    )
    return parser.parse_args()


args = parse_args()
model_input_size = args.model_input_size

model = ResNet(args.model_arch)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# model.cuda()

batch_size = 1
x = torch.randn(batch_size, 3, model_input_size, model_input_size, requires_grad=True)
torch_out = model(x)

# Export the model
torch.onnx.export(
    model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    re.sub(
        "pth", "onnx", args.model_path
    ),  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
)
