import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for key in ['audio_dir', 'frames_dir', 'annotation_dir', 'output_dir', 'log_dir']:
        if key in config['data']:
            config['data'][key] = os.path.abspath(config['data'][key])
    config['logging']['log_dir'] = os.path.abspath(config['logging']['log_dir'])
    return config
