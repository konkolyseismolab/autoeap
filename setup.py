from setuptools import setup
from os import path
import sys

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

desc='Automated version of Extended Aperture Photometry developed for K2 RR Lyrae stars.'

setup(name='autoeap',
      version=__version__,
      description=desc,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Pal Szabo',
      author_email='ps738@cam.ac.uk',
      url='https://github.com/zabop/autoeap/',
      package_dir={'autoeap':'src'},
      packages=['autoeap'],
      install_requires=requirements,
      entry_points=entry_points
     )
