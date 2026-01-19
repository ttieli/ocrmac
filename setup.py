#!/usr/bin/env python

"""The setup script."""

from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent

readme = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""
history = (here / "HISTORY.md").read_text(encoding="utf-8") if (here / "HISTORY.md").exists() else ""

requirements = [
    "Click>=7.0",
    "pyobjc-framework-Vision",
    "pillow",
    "requests",
    "PyMuPDF>=1.24.0",
    "python-docx",
    "numpy",  # Required for adaptive OCR preprocessing
    "opencv-python>=4.5.0",  # Required for region detection
]

# Optional dependencies
extras_require = {}

test_requirements = [
    "pytest>=3",
]

setup(
    author="Maximilian Strauss",
    author_email="straussmaximilian@gmail.com",
    python_requires=">=3.9",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    description="A python wrapper to extract text from images on a mac system. Uses the vision framework from Apple.",
    entry_points={
        "console_scripts": [
            "ocrmac=ocrmac.cli:main",
            "macocr=ocrmac.cli:main",  # Alias command
        ],
    },
    install_requires=requirements,
    extras_require=extras_require,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="ocrmac",
    name="ocrmac",
    packages=find_packages(include=["ocrmac", "ocrmac.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ttieli/ocrmac",
    version="1.3.0",
    zip_safe=False,
)
