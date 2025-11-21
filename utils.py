import yaml
class Hypernum:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as f:
            params = yaml.safe_load(f)
        return cls(**params)