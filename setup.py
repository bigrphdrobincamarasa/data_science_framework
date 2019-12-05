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
command = 'git --git-dir {}/.git rev-parse HEAD'.format(
    os.path.dirname(os.path.abspath(__file__))
)
with os.popen(cmd=command) as stream:
      version = stream.read()[:-1]

# Get requirements
with open('requirements.txt', 'r') as requirements_handler:
    requirements = [
        dependency
        for dependency in requirements_handler.readlines()
        if not 'data_science_framework' in dependency
    ]

setup(
    name='data_science_framework',
    author='Robin Camarasa',
    version=version,
    packages=find_packages(),
    description='data\_science\_framework is a tiny library, that allows you to develop' + \
              ' your own customable datascience project',
    long_description=long_description,
    install_requires=requirements,
    author_email='r.camarasa@erasmusmc.nl',
)

