from setuptools import setup, find_packages
from catlearn import __version__

DESCRIPTION = "Machine Learning using atomic-scale calculations"
LONG_DESCRIPTION = "Machine Learning using atomic-scale calculations"

# Setting up
setup(
    name="catlearn",
    version=__version__,
    author="Andreas Vishart",
    author_email="<alyvi@dtu.dk>",
    url="https://github.com/avishart/CatLearn/tree/restructuring",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.3", "scipy>=1.8.0", "ase>=3.22.1"],
    extras_require={
        "optional": ["mpi4py>=3.0.3", "dscribe>=2.1", "matplotlib>=3.8"]
    },
    test_suite="tests",
    tests_require=["unittest"],
    keywords=["python", "gaussian process", "machine learning", "regression"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
)
