import os

import gdown


def check_dir(dir_name: str) -> bool:
    if os.path.isdir(dir_name):
        return True
    return False


def download_tfrecords(dir_name="tfrecords") -> None:
    if not check_dir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    gdown.download(
        "https://drive.google.com/uc?id=1jBHxybNWx4uhLxWxyguHpmjzz73YMQdG", quiet=False
    )
    os.system("tar -xf tfrecords.tar.gz")
    os.remove("tfrecords.tar.gz")
