from setuptools import setup, find_packages
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('spiralfilm/__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

setup(
    name="SpiralFilm",
    version=version,
    url="https://github.com/Spiral-AI/spiralfilm",
    author="Yuichi Sasaki",
    author_email="y_sasaki@go-spiral.ai",
    description="A thin wrapper for the OpenAI GPT family of APIs",
    packages=find_packages(),
    install_requires=["openai", "tiktoken"],
    python_requires=">=3.8, <4",
    keywords="openai, gpt, api, wrapper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
