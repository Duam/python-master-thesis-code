## Master Thesis Code
This repository holds the code used in my master thesis.

Title: "Modeling, Identification, Estimation and Control of a Rotational Startup System for Airborne Wind Energy"

Author: Paul Philipp Quirin Daum

University: Albert-Ludwigs-Universität Freiburg, Faculty of Engineering, Department of Microsystems Engineering

Examiner: Prof. Dr. Moritz Diehl, Prof. Dr.-Ing. Jörg Fischer

Adviser: Jochem de Schutter

Date: November 13th, 2019

## How to setup

```bash
# Make sure you have the dependencies
sudo apt install python-dev

# -- Install ZeroCM --
# ZeroCM is used for communication between the experimental setup
# (the "carousel") and the experimenting computer.

# Initialize ZeroCM submodule
git submodule update --init
cd extern/zerocm

# See extern/zerocm/README.md for a description of the following commands
./scripts/install_deps.sh
./waf configure --python=/usr/bin/python3.8 --use-python --use-udpm
./waf build
sudo ./waf install

# Go back home
cd ../../

# OPTIONAL: (the output files are included in the repo)
	# Using zerocm, create the message definition
	cd src/thesis_code/zcm_message_definitions/
	zcm-gen -p timestamped_vector_double.zcm
	zcm-gen -p timestamped_vector_float.zcm


# -- Install python package --
virtualenv --system-site-packages venv
source ./venv/bin/activate
pip install -e .

```

## Disclaimer

I'm currently cleaning up this repository.
