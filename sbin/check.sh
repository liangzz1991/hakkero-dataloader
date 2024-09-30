#!/usr/bin/env bash
#

SELF_DIR=$(cd $(dirname ${BASH_SOURCE});pwd)

PROJECT_ROOT=${SELF_DIR}/..

# ----------------------------------------------------------------------------------------------------------------------
check_fmt() {
  cd ${PROJECT_ROOT}

  ${ROOT_PYTHON_BIN:+${ROOT_PYTHON_BIN}/}isort hakkero || exit 1
  ${ROOT_PYTHON_BIN:+${ROOT_PYTHON_BIN}/}autopep8  hakkero || exit 1
  ${ROOT_PYTHON_BIN:+${ROOT_PYTHON_BIN}/}black hakkero || exit 1

  cd ${SELF_DIR}
}


# ----------------------------------------------------------------------------------------------------------------------
check_fmt