from os import path
import sys
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

sys.path.insert(0, "src")
from version import __version__

# Load requirements
requirements = None
with open('requirements.txt') as file:
    requirements = file.read().splitlines()

# If Python3: Add "README.md" to setup.
# Useful for PyPI. Irrelevant for users using Python2.
try:
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

# Command-line tools
entry_points = {'console_scripts': [
    'autoeap = autoeap:autoeap_from_commandline'
]}

extensions = Extension("PDM",
                         sources=["src/PDM/PDM_pyc.pyx", "src/PDM/pdm.c"],
                         include_dirs=[numpy.get_include(),'/opt/local/include'],
                         library_dirs=['/opt/local/lib'],
                         libraries=['m'],
                         define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

desc='Automated version of Extended Aperture Photometry developed for high amplitude K2 variable stars.'

setup(name='autoeap',
      version=__version__,
      description=desc,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Attila Bodi & Pal Szabo',
      author_email='bodi.attila@csfk.org',
      url='https://github.com/konkolyseismolab/autoeap/',
      package_dir={'autoeap':'src'},
      packages=['autoeap'],
      install_requires=requirements,
      entry_points=entry_points,
      ext_modules = cythonize(extensions, compiler_directives={'language_level': 3})
     )
