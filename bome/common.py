import os
import sys
from logging import Logger
import yaml
sys.path.append(".")


def basicOption():
    return {
        "INLINE": {"-on": [], "-off": [], "-recursive": []},
        "BALANCE": {"-on": [], "-off": []},
        "FLATTEN": {"-on": [], "-off": []},
        "PIPELINE": {"-style": []},
        "UNROLL": {"-factor": []},
        "PARTITION": {"-factor": [], "-type": []},
        "RESHAPE": {"-factor": [], "-type": []},
        "STORAGE": {"-type": [], "-impl": [], "-latency": []},
    }


def createFolder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def getYaml(yaml_file, log: Logger=None):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    if log:
        log.info(file_data)
    else:
        print(file_data)
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data
