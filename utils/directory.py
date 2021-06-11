import pathlib


def make_validation_directories(training_directory_name, epoch):
    make_validation_dir(training_directory_name, epoch)
    for data_type in ["gt", "detections"]:
        make_validation_dir(training_directory_name, epoch, data_type)


def make_validation_dir(training_directory_name, epoch, datatype="gt"):
    if datatype == "":
        (training_directory_name / pathlib.Path(f"epoch_{epoch}")).mkdir(
            parents=True, exist_ok=True
        )
    else:
        (training_directory_name / pathlib.Path(f"epoch_{epoch}/{datatype}")).mkdir(
            parents=True, exist_ok=True
        )
