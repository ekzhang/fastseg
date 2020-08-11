"""Setup"""

import pathlib
from setuptools import setup, find_namespace_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="fastseg",
    version="0.0.1",
    description="Fast Semantic Segmentation for PyTorch",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ekzhang/fast-semantic-seg",
    author="Eric Zhang",
    author_email="ekzhang1@rgmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Graphics",
    ],
    packages=find_namespace_packages(include=["fastseg", "fastseg.*"]),
    include_package_data=True,
    install_requires=[
        "geffnet >= 0.9.8",
        "Pillow >= 7.0.0",
        "numpy >= 1.18.0",
        "torch >= 1.6.0",
        "torchvision >= 0.6.0",
    ],
)
