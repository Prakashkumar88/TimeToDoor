# Auto-generated file.
from setuptools import setup, find_packages
from typing import List

PROJECT_NAME = "TimeToDoor"
VERSION = "0.0.1"
AUTHOR = "Prakash Kumar"
AUTHOR_EMAIL = "dummymail@ml.com"
DESCRIPTION = "A project to predict time to door using machine learning."

REQUIREMENTS_FILE = "requirements.txt"

HYPHEN_E_DOT = "-e ."

def get_requirements() -> List[str]:
    with open(REQUIREMENTS_FILE) as requirements_file:
        requirements = requirements_file.readlines()
        requirements = [requirement_name.replace("\n", "") for requirement_name in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
        return requirements

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires='>=3.6',
)