#!/usr/bin/env bash
set -e # Abort on error
echo "npm (Node Package Manager) must be installed to build the ui. See https://www.npmjs.com/get-npm"
echo "Building ui"
cd ui 

# CLEAN
rm -rf node_modules
rm -rf dist

npm install && npm run build

touch dist/__init__.py
touch dist/WARNING__THIS_CODE_IS_GENERATED
echo "WARNING: Do NOT edit files in this directory! This code is generated with npm run build by build_ui.sh and all files are replaced when building. Instead, make changes in the../ui" > dist/README.md

# Move to target directory
cd ../
rm -rf pistarlab/uidist
mv ui/dist pistarlab/uidist
echo "UI Build Complete"