# Installation

GIPPY can be installed directly from PyPi using pip, or can be installed from a clone of the repository.
There are a few dependencies that must be installed first. These notes are for Ubuntu.

    1) Install the UbuntuGIS Repository:
    $ sudo apt-get install python-software-properties
    $ sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
    $ sudo apt-get update

    2) Installed required dependencies
    $ sudo apt-get install python-dev python-setuptools python-numpy python-gdal g++ libgdal1-dev gdal-bin libboost-all-dev swig2.0 swig

    3) Install pip (if not installed)
    $ sudo easy_install pip

    4) Install GIPPY (sudo not required if installing to virtual environment)
    $ sudo pip install gippy
    -or-
    $ git clone http://github.com/matthewhanson/gippy.git
    $ cd gippy
    $ sudo ./setup.py install