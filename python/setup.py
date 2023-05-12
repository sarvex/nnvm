import os
import sys
from setuptools import find_packages
from distutils.core import setup

def config_cython():
    # temporary disable cython for now
    # as NNVM uses local DLL build
    return []

# We can not import `libinfo.py` in setup.py directly since __init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, './nnvm/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
_, LIB_NAME = os.path.split(LIB_PATH[0])
__version__ = libinfo['__version__']
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
rpath = os.path.relpath(LIB_PATH[0], curr_path)

setup(name='nnvm',
      version=__version__,      
      description="NNVM: Open Compiler for AI Frameworks",
      zip_safe=False,
      install_requires=[
        'numpy'
      ],
      packages=find_packages(),
      url='https://github.com/dmlc/nnvm',
      include_package_data=True,
      data_files=[('nnvm', [rpath])])

