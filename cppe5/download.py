import os

import gdown


def check_dir(dir_name: str) -> bool:
    if os.path.isdir(dir_name):
        return True
    return False


def download_data(dir_name="data") -> None:
    if not check_dir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    gdown.download(
        "https://drive.google.com/uc?id=1MGnaAfbckUmigGUvihz7uiHGC6rBIbvr", quiet=False
    )
    os.system("tar -xf dataset.tar.gz")
    os.remove("dataset.tar.gz")
