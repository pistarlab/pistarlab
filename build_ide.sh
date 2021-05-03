#!/usr/bin/env bash
set -e # Abort on error

echo "npm (Node Package Manager) must be installed to build the ui. See https://www.npmjs.com/get-npm"

echo "Building theia_ide"
cd theia_ide 
rm -rf node_modules
yarn && yarn theia build
echo "Done building IDE"
