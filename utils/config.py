import yaml


class Config:
    def __init__(self, config_path):
        self.settings = self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
