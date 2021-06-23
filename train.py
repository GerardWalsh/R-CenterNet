import argparse
from datetime import datetime
import pathlib
import time
import os
import sys

import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from opencv_transforms import transforms

from dataset import ctDataset
from Loss import CtdetLoss
from predict import (
    post_process_model_outs,
    post_process_sample_outs,
    post_process,
    merge_outputs,
    final_output,
)
from utils.yaml import read_yaml, write_yaml
from utils.directory import make_validation_directories
from utils.boxes import dump_validation_batch

sys.path.append(r"./backbone")
from resnet import ResNet


image_output_meta = {
    "c": np.array([960.0, 540.0], dtype=np.float32),
    "s": 1920.0,
    "out_height": 56,
    "out_width": 56,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-config",
        type=str,
        required=True,
        help="Name of the model arch.",
    )
    return parser.parse_args()


args = parse_args()
config = read_yaml(args.training_config)
date_stamp = datetime.now().strftime("%j:%H:%M")


opts = {
    "RMSprop": torch.optim.RMSprop,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()
model = ResNet(int(config["model"].split("_")[-1]), head_conv=int(config["head_conv"]))

loss_weight = {"hm_weight": 1, "wh_weight": 0.1, "ang_weight": 0.5, "reg_weight": 0.1}
criterion = CtdetLoss(loss_weight)

device = torch.device("cuda")
if use_gpu:
    model.cuda()

model.train()

learning_rate = float(config["learning_rate"])
num_epochs = int(config["epochs"])

# different learning rate
params = []
params_dict = dict(model.named_parameters())
for key, value in params_dict.items():
    params += [{"params": [value], "lr": learning_rate}]

print(f'[INFO]: Using {str(config["optimizer"])} optimiser.')
optimizer = opts[str(config["optimizer"])](params, lr=learning_rate)

transform = transforms.Compose(
    [
        transforms.RandomGrayscale(),
        # # transforms.ToTensor(),
        # torch.from_numpy,
        # transforms.ToPILImage(),
        # transforms.ColorJitter(hue=.5, saturation=.5)
        # transforms.ToTensor()
    ]
)

# Setup hyperparameters
if config["center"]:
    centered = "centered"
else:
    centered = ""

train_dataset = ctDataset(
    split="train",
    transform=transform,
    input_size=int(config["input_size"]),
    center=bool(config["center"]),
)
train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
)

test_dataset = ctDataset(
    split="val",
    input_size=int(config["input_size"]),
    center=bool(config["center"]),
)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
print("[INFO]: The dataset has %d images" % (len(train_dataset)))


num_iter = 0
best_test_loss = np.inf
start = time.perf_counter()

# Create directory for training run
training_directory_name = pathlib.Path(f"{date_stamp}")
training_directory_name.mkdir(parents=True, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    if epoch == 90:
        learning_rate = learning_rate * 0.1
    if epoch == 120:
        learning_rate = learning_rate * (0.1 ** 2)
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    total_loss = 0.0

    for i, sample in enumerate(train_loader):

        for k in sample:
            if k != "filepath":
                sample[k] = sample[k].to(device=device, non_blocking=True)

        pred = model(sample["input"])
        loss = criterion(pred, sample)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 30 == 0:
            print(
                "Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f"
                % (
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    len(train_loader),
                    loss.data,
                    total_loss / (i + 1),
                )
            )
            num_iter += 1

    # validation
    validation_loss = 0.0
    model.eval()

    # create directories for storing validation results
    make_validation_directories(training_directory_name, epoch)

    for i, sample in enumerate(test_loader):
        if use_gpu:
            for k in sample:
                if k != "filepath":
                    sample[k] = sample[k].to(device=device, non_blocking=True)

        pred = model(sample["input"])

        # Post process output to calculate map
        pred["hm"] = pred["hm"].sigmoid_()
        model_detections = post_process_model_outs(pred, image_output_meta)
        sample_detections = post_process_sample_outs(sample, image_output_meta)
        sample_image_filepaths = sample["filepath"]
        dump_validation_batch(
            training_directory_name,
            epoch,
            sample_detections,
            model_detections,
            sample_image_filepaths,
        )
        loss = criterion(pred, sample)
        validation_loss += loss.item()
    validation_loss /= len(test_loader)

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print("Got best test loss %.5f" % best_test_loss)
        torch.save(
            model.state_dict(), training_directory_name / f"epoch_{epoch}_best.pth"
        )

    torch.save(model.state_dict(), training_directory_name / "last.pth")

print(
    "{} epochs with {} frames in {} seconds".format(
        num_epochs, len(train_dataset), time.perf_counter() - start
    )
)

print("Saving config")
config["datestamp"] = date_stamp
write_yaml(training_directory_name / "trained_config.yml", config)
