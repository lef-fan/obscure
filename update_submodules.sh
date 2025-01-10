#!/bin/bash
SUBMODULE_URL="https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip"
SUBMODULE_DIR="libs/"

curl -L -o libtorch.zip $SUBMODULE_URL
unzip -o libtorch.zip -d $SUBMODULE_DIR
rm libtorch.zip