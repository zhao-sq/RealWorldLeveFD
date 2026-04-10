import os

from setuptools import find_packages, setup

with open("version.txt", "r") as file_handler:
    __version__ = file_handler.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="crx_rmi_utils",
    description="CRX RMI Drive Utils.",
    author="Yu Zhao, Hsien-Chung Lin",
    author_email="yu.zhao@fanucamerica.com, hsien-chung.lin@fanucamerica.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={"rl_forcecontrol": ["version.txt"]},
    version=__version__,
    install_requires=["numpy",
                      "scipy",
                      "matplotlib",
                      'pygame',
                      ],
    extras_require={
        "tests": ["pytest", "black", "pytype"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)