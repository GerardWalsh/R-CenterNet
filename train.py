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
from utils.yaml import read_yaml, write_yaml

sys.path.append(r"./backbone")
from resnet import ResNet

# from resnet_dcn import ResNet
# from dlanet import DlaNet
# from dlanet_dcn import DlaNet
# import predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-config",
        type=str,
        required=True,
        help="Name of the model arch.",
    )
    # parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate.")
    # parser.add_argument("--input-size", type=int, default=224, help="Image input size.")
    # parser.add_argument(
    #     "--center",
    #     type=bool,
    #     default=False,
    #     help="Whether to center the data by the mean and std dev.",
    # )
    # parser.add_argument(
    #     "--epochs",
    #     type=int,
    #     default=10,
    #     help="Whether data has been normalized.",
    # )
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     default=4,
    #     help="Whether data has been normalized.",
    # )
    # parser.add_argument(
    #     "--optimiser",
    #     type=str,
    #     default="SGD",
    #     help="Which optimiser to use.",
    # )
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
model = ResNet(int(config["model"].split("_")[-1]))


loss_weight = {"hm_weight": 1, "wh_weight": 0.1, "ang_weight": 0.5, "reg_weight": 0.1}
criterion = CtdetLoss(loss_weight)

device = torch.device("cuda")
if use_gpu:
    model.cuda()

# summary(model, input_size=(3, 512, 512))

model.train()

learning_rate = float(config["learning_rate"])
num_epochs = int(config["epochs"])

# different learning rate
params = []
params_dict = dict(model.named_parameters())
for key, value in params_dict.items():
    params += [{"params": [value], "lr": learning_rate}]

# optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
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
# train_dataset = ctDataset(split='train')
train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
)

test_dataset = ctDataset(split="val")
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
print("[INFO]: The dataset has %d images" % (len(train_dataset)))


num_iter = 0
best_test_loss = np.inf
start = time.perf_counter()

# Create directory for training run
training_directory_name = pathlib.Path(
    f'{date_stamp}_{config["model"]}_{config["input_size"]}_epochs:{config["epochs"]}_lr:{config["learning_rate"]}_{config["optimiser"]}_{centered}_batch_size:{config["batch_size"]}'
)
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
    for i, sample in enumerate(test_loader):
        if use_gpu:
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)

        pred = model(sample["input"])
        pred["hm"] = pred["hm"].sigmoid_()
        loss = criterion(pred, sample)
        validation_loss += loss.item()
    validation_loss /= len(test_loader)

    # if best_test_loss > validation_loss:
    #     best_test_loss = validation_loss
    #     print('Got best test loss %.5f' % best_test_loss)
    #     torch.save(model.state_dict(),'best.pth')

    torch.save(
        model.state_dict(),
        training_directory_name / "last.pth",
    )

print(
    "{} epochs with {} frames in {} seconds".format(
        num_epochs, len(train_dataset), time.perf_counter() - start
    )
)

print("Saving config")
config["datestamp"] = date_stamp
write_yaml(training_directory_name / "trained_config.yml", config)
