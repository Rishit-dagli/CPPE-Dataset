import json
import os

from PIL import Image

num_test_image = 29
num_train_image = 1000


def convert_png2jpg_image(png_file: str, jpg_file: str) -> None:
    img = Image.open(png_file)
    img = img.convert("RGB")
    img.save(jpg_file, "JPEG")


def convert_png2jpg(
    png_dir: str, jpg_dir: str, num_images: int, annotation_file: str
) -> None:
    for i in range(1, num_images + 1):
        data = json.load(open(annotation_file))
        data["images"][i - 1]["file_name"] = (
            data["images"][i - 1]["file_name"][:-3] + ".jpg"
        )
        with open(annotation_file, "w") as f:
            json.dump(data, f)

        png_file = png_dir + data["images"][i - 1]["file_name"]
        jpg_file = jpg_dir + data["images"][i - 1]["file_name"][:-3] + ".jpg"
        convert_png2jpg_image(png_file, jpg_file)
        os.remove(png_file)


def convert_png2jpg_directory() -> None:
    convert_png2jpg(
        png_dir="data/images/",
        jpg_dir="data/images/",
        num_images=num_train_image,
        annotation_file="data/annotations/train.json",
    )
    convert_png2jpg(
        png_dir="data/images/",
        jpg_dir="data/images/",
        num_images=num_test_image,
        annotation_file="data/annotations/test.json",
    )
