import sys

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

sys.path.append(r"./backbone")

from resnet import ResNet
from dlanet import DlaNet

model = ResNet(18)
model.load_state_dict(
    torch.load(
        "./last_resnet_18_224_epochs:100_lr:0.002_RMSprop_centered_batch_size:6.pth"
    )
)
model.eval()
# model.cuda()

batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = model(x)

# Export the model
torch.onnx.export(
    model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    "w_centernet_resnet_18_224_epochs:100_lr:0.002_RMSprop_centered_batch_size:6.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
)
