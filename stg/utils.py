import yaml

def read_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)
