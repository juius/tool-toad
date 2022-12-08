#!/usr/bin/env python

import os

import setuptools

__version__ = "0.1"

# Find the absolute path
here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, "README.rst")) as f:
    long_description = f.read()

short_description = "A collection of computational chemisty utilities."


setuptools.setup(
    name="tool_toad",
    version=__version__,
    maintainer="Juls",
    description=short_description,
    long_description=long_description,
    url="https://github.com/juius/tool_toad",
    python_requires=">=3.9",
)
