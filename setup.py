"""
Setup configuration for RAG-Lite package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().splitlines() 
        if line.strip() and not line.startswith("#")
    ]

# Core requirements (without optional evaluation dependencies)
core_requirements = [
    "ollama>=0.1.0",
    "chromadb>=1.0.0",
]

# Optional evaluation requirements
eval_requirements = [
    "datasets>=2.14.0",
]

setup(
    name="rag-lite",
    version="0.1.0",
    description="A light RAG implementation for knowledge-based question answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Eddie J.",
    author_email="jinxingyu95@gmail.com",
    url="https://github.com/XingyuJinTI/rag-lite",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=core_requirements,
    extras_require={
        "eval": eval_requirements,
        "all": eval_requirements,
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "rag-lite=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
