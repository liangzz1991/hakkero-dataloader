#!/usr/bin/env bash

SELF_DIR=$(cd $(dirname ${BASH_SOURCE});pwd)

autoflake ${SELF_DIR}/../hakkero
isort ${SELF_DIR}/../hakkero
autopep8 ${SELF_DIR}/../hakkero
black ${SELF_DIR}/../hakkero


autoflake ${SELF_DIR}/../setup.py
isort ${SELF_DIR}/../setup.py
autopep8 ${SELF_DIR}/../setup.py
black ${SELF_DIR}/../setup.py