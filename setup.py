"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-25

**Project** : data_science_framework

**  **
"""
import os
import sys
from subprocess import check_output

from setuptools import find_packages, setup

# Get long description
with open('README.md', 'r') as readme_handler:
      long_description = readme_handler.read()

# Get version
out = check_output(["git", "branch"]).decode("utf8")
current = next(line for line in out.split("\n") if line.startswith("*"))
version = current.strip("*").strip().replace('-', '.')

# Get requirements
with open('requirements.txt', 'r') as requirements_handler:
      requirements = requirements_handler.readlines()

setup(
      name='data_science_framework',
      version=version,
      packages=find_packages(),
      description='data\_science\_framework is a tiny library, that allows you to develop' + \
                  ' your own customable datascience project',
      long_description=long_description,
      install_requires=requirements,
      author_email='r.camarasa@erasmusmc.nl',
)

