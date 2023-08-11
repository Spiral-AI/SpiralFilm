#!/bin/zsh

rm -rf SpiralFilm.egg-info/*
rm -rf dist build
rm -rf spiralfilm/__pycache__
rm -rf examples/.cache.pickle

python setup.py sdist
python setup.py bdist_wheel

twine upload --repository pypi dist/*