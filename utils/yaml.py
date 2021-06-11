import yaml


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc


def write_yaml(yaml_path: str, yaml_data: dict) -> dict:
    with open(yaml_path, "w") as stream:
        yaml.safe_dump(yaml_data, stream, default_flow_style=False)
