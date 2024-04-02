# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

from setuptools import setup, find_packages

__version__ = '1.0.0'

with open("README.md", "r", encoding="utf-8") as fid:
  long_description = fid.read()

setup(
    name='ognn',
    version=__version__,
    author='Peng-Shuai Wang',
    author_email='wangps@hotmail.com',
    url='https://github.com/octree-nn/ognn-pytorch',
    description='Dual Octree Graph Neural Networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['torch', 'numpy', 'ocnn', 'easydict'],
    python_requires='>=3.6',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
