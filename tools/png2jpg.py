import argparse
import json
import os

from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default",
        const=True,
        nargs="?",
        type=bool,
        help="Use the default setting and paths to convert png to jpg",
    )
    parser.add_argument(
        "--png_dir",
        type=str,
        help="Path to the directory containing png images",
    )
    parser.add_argument(
        "--jpg_dir",
        type=str,
        help="Path to the directory to save jpg images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        help="Number of images to convert",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        help="Path to the annotation file",
    )
    return parser.parse_args()


def convert_png2jpg_image(png_file: str, jpg_file: str) -> None:
    img = Image.open(png_file)
    img = img.convert("RGB")
    img.save(jpg_file, "JPEG")


def convert_png2jpg(
    png_dir: str, jpg_dir: str, num_images: int, annotation_file: str
) -> None:
    for i in tqdm(range(1, num_images + 1)):
        data = json.load(open(annotation_file))
        png_file = png_dir + data["images"][i - 1]["file_name"]
        jpg_file = jpg_dir + data["images"][i - 1]["file_name"][:-4] + ".jpg"

        data["images"][i - 1]["file_name"] = (
            data["images"][i - 1]["file_name"][:-4] + ".jpg"
        )
        with open(annotation_file, "w") as f:
            json.dump(data, f)

        convert_png2jpg_image(png_file, jpg_file)
        os.remove(png_file)


def convert_png2jpg_directory() -> None:
    num_test_image = 29
    num_train_image = 1000
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
    print()
    print(
        f"Converted {num_train_image+num_test_image} images from data/images/ to data/images/"
    )
    print(f"The annotations data/annotations/ have been updated")


if __name__ == "__main__":
    args = parse_args()
    if args.default:
        print(
            "Note: This uses default paths, so your image directory should be data/images/ and the annotation files should be data/annotations/train.json and data/annotations/test.json"
        )
        print()
        convert_png2jpg_directory()
    else:
        convert_png2jpg(
            png_dir=args.png_dir,
            jpg_dir=args.jpg_dir,
            num_images=args.num_images,
            annotation_file=args.annotation_file,
        )
        print()
        print(
            f"Converted {args.num_images} images from {args.png_dir} to {args.jpg_dir}"
        )
        print(f"The annotations {args.annotation_file} have been updated")
