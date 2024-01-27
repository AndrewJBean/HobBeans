#!/bin/bash

# User should source this file to set up the environment

DIR_OF_THIS_FILE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# set PYTHONPATH to include DIR_OF_THIS_FILE
# only if it is not already set
if [[ ":$PYTHONPATH:" != *":$DIR_OF_THIS_FILE:"* ]]; then
    export PYTHONPATH="${PYTHONPATH}:$DIR_OF_THIS_FILE"
    echo "adding $DIR_OF_THIS_FILE to PYTHONPATH"
    echo "PYTHONPATH: $PYTHONPATH"
else
    echo "PYTHONPATH already contains $DIR_OF_THIS_FILE"
fi
