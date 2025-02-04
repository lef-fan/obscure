#!/bin/bash
SUBMODULE_URL="https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip"
SUBMODULE_ROOTDIR="libs/"
SUBMODULE_DIR="libs/libtorch"

echo "Updating modules from $MODULE_URL..."

if [ -d "$SUBMODULE_DIR" ]; then
    rm -rf "$SUBMODULE_DIR"
fi

echo "Downloading libtorch module..."
curl -L -o libtorch.zip $SUBMODULE_URL

echo "Extracting libtorch module zip..."
unzip -o libtorch.zip -d $SUBMODULE_ROOTDIR

rm libtorch.zip

echo "Modules updated successfully."