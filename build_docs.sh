#!/usr/bin/env bash
set -e # Abort on error
echo "Building Documentation and Making Available through local instance"


pip install -r docs/requirements.txt

cd docs
sphinx-apidoc -o source ../pistarlab
make html

cd ../
rm -rf pistarlab/doc_dist
cp -R docs/build/html pistarlab/doc_dist

touch pistarlab/doc_dist/__init__.py
touch pistarlab/doc_dist/WARNING__THIS_CODE_IS_GENERATED
echo "Doc Build Complete"