from setuptools import setup, find_packages

setup(
    name="SpiralFilm",
    version="0.1.0",
    url="https://github.com/Spiral-AI/spiralfilm",
    author="Yuichi Sasaki",
    author_email="y_sasaki@go-spiral.ai",
    description="A thin wrapper for the OpenAI GPT family of APIs",
    packages=find_packages(),
    install_requires=["openai", "tiktoken"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
