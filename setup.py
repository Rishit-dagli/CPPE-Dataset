import os.path

from setuptools import find_packages, setup


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cppe5",
    version=get_version("cppe5/version.py"),
    description="A library to easily download, load and work with the CPPE-5 dataset.",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    url="https://github.com/Rishit-dagli/CPPE-Dataset",
    author="Rishit Dagli, Ali Mustufa",
    author_email="rishit.dagli@gmail.com",
    install_requires=["gdown ~= 4.2.0", "pycocotools ~= 2.0.3"],
    extras_require={
        "dev": [
            "check-manifest",
            "twine",
            "black",
        ],
    },
)
