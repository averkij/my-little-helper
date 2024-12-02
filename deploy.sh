#!/bin/sh

# python -m pip install --upgrade build
# python -m pip install --user --upgrade twine

python -m build
python -m twine upload dist/*