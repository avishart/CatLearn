from setuptools import setup, find_packages

VERSION = '5.0.8' 
DESCRIPTION = 'Machine Learning using atomic-scale calculations'
LONG_DESCRIPTION = 'Machine Learning using atomic-scale calculations'

# Setting up
setup(name="catlearn", 
      version=VERSION,
      author="Andreas Vishart",
      author_email="<alyvi@dtu.dk>",
      url="https://github.com/avishart/CatLearn/tree/restructuring",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      packages=find_packages(),
      install_requires=['numpy>=1.20.3','scipy>=1.8.0','ase>=3.22.1','mpi4py>=3.0.3'], 
      python_requires='>=3.7',
      test_suite='tests',
      tests_require=['unittest'],
      keywords=['python','gaussian process','machine learning','regression'],
      classifiers=["Development Status :: 4 - Beta",
                   "Intended Audience :: Education",
                   "Programming Language :: Python :: 3",
                   "Operating System :: MacOS :: MacOS X",
                   "Operating System :: POSIX :: Linux"]
      )
