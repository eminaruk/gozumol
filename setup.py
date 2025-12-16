#!/usr/bin/env python3
"""
Gozumol - Visual Navigation Assistant for Visually Impaired Users
Setup script for package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="gozumol",
    version="0.1.0",
    author="Gozumol Team",
    description="AI-powered visual navigation assistant for visually impaired users",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eminaruk/gozumol",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch==2.6.0",
        "flash_attn==2.7.4.post1",
        "transformers==4.48.2",
        "accelerate==1.3.0",
        "pillow==11.1.0",
        "numpy>=1.24.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "qwen": [
            "qwen-vl-utils",
        ],
        "camera": [
            "opencv-python>=4.8.0",
        ],
        "tts": [
            "pyttsx3>=2.90",
            "gTTS>=2.3.0",
        ],
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gozumol=gozumol.cli:main",
        ],
    },
    include_package_data=True,
    keywords="vision, accessibility, navigation, multimodal, AI, visually-impaired",
)
