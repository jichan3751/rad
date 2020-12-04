#!/bin/bash

# DEST: end with '/'
DEST="$HOME/pp/tmp/201204_tmp/"

# sync to destination dir
mkdir -p ${DEST}
rsync -a --delete $(pwd)/ ${DEST}

# move and run script
pushd ${DEST}
bash docker_run.sh
popd

# rsync -a --delete ${DEST}/ $(pwd)/
rsync -a ${DEST} $(pwd)/
