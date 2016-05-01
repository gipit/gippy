Installation
++++++++++++

Gippy is a C++ library with Python bindings. The C++ portion, libgip.so and C++ wrappers wrappers around it, are built as extensions using Python setuptools, so do not require seperate installation. Gippy has been designed to use as few system dependencies as possible, however there are a few.s


On Ubuntu::

    # if not already added, add the UbuntuGIS repository
    $ sudo apt-get install python-software-properties
    $ sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
    $ sudo apt-get update

    # install dependencies
    $ sudo apt-get install libgdal1h gdal-bin libgdal-dev g++4.8
    # install pip if not installed
    # sudo easy_install pip

On OS X (using brew)::

    $ brew install 


With the dependencies met, gippy can be installed via pip from it's repository on PyPi. If installing to a `virtual environment <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_, activate the environment first. If installing systeym-wide pip will need to be run as sudo.

    $ pip install gippy


Development Installation
========================

For development purposes, the swig wrappers must be regenerated anytime the C++ code is modified, and thus the swig package (currently using swig2.0) must be installed.

On Ubuntu::

    $ sudo apt-get install swig

On OS X::

    $ brew install swig

Then install gippy as a development installation by cloing the repository. Links will be installed in the Python packages directory that point to the directory where gippy resides.

.. code::

    $ git clone http://github.com/gipit/gippy.git
    $ cd gippy
    $ pip install -e .


Docs
====

To generate docs:

.. code::

    $ cd docs
    $ make html

Open `docs/_build/html/index.html` in the browser


Testing
=======

Gippy testing is done on the Python siode sing the nosetest testing framework and the `sat-testdata <https://github.com/sat-utils/sat-testdata>`_ repository for test imagery, which is installed as a requirement. Run the tests from the test directory.

.. code::

    $ cd test
    $ nosetests

