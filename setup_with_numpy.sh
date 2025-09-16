#!/bin/bash
# Custom setup script that includes numpy environment variables

# Set numpy include paths for compilation
export CPATH="/usr/lib/python3/dist-packages/numpy/core/include:$CPATH"
export C_INCLUDE_PATH="/usr/lib/python3/dist-packages/numpy/core/include:$C_INCLUDE_PATH"
export CXX_INCLUDE_PATH="/usr/lib/python3/dist-packages/numpy/core/include:$CXX_INCLUDE_PATH"

# Source the original setup.sh
source install/setup.sh

