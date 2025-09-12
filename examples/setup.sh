#!/bin/bash

conda create --prefix ./env -y python=3.9

./env/bin/pip install -r examples/requirements.txt