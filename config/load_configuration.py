import os
import glob
import yaml


def _load_cfg(cfg_path):
    cfg = {}
    if cfg_path is not None:
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
    return cfg


class LoadConfiguration(object):
    _instance = None

    @classmethod
    def _load_configuration(cls):
        config = {}

        globbed = glob.glob(
            os.path.join(
                os.path.dirname(__file__), "*.yaml"
            )
        )

        for file_path in globbed:
            config_file_name = os.path.split(file_path)[1]
            name, extension = config_file_name.split('.')

            if name == 'config':
                config.update(_load_cfg(file_path))
            else:
                config[name] = _load_cfg(file_path)

        cls._instance = config

    def __new__(cls):
        if cls._instance is None:
            cls._load_configuration()

        return cls._instance