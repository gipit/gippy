Installation
++++++++++++

Gippy is a C++ library with Python bindings. The C++ portion, libgip.so and C++ wrappers wrappers around it, are built as extensions using Python setuptools, so do not require seperate installation. Gippy has been designed to use as few system dependencies as possible, however there are a few.s


On Ubuntu (14.04)::

    $ sudo apt-get install libgdal-dev python-setuptools g++ python-dev
    # sudo easy_install pip             # if pip not already installed
    $ sudo pip install numpy            # pre-install numpy

On OS X (using brew)::

    $ brew install gdal


With the dependencies met, gippy can be installed via pip from it's repository on PyPi. If installing to a `virtual environment <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_, activate the environment first. If installing system-wide pip will need to be run as sudo.

.. code::

    $ pip install gippy --pre

To install a beta version use the --pre switch. Without --pre, pip will install the last release version, which is currently 0.3.5.


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

