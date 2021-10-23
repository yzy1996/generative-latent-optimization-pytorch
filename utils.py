import os
import yaml

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_config(data, path):
    with open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)