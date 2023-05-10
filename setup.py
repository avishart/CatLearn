from setuptools import setup, find_packages

VERSION = '3.2.2' 
DESCRIPTION = 'Machine Learning using atomic-scale calculations'
LONG_DESCRIPTION = 'Machine Learning using atomic-scale calculations'

# Setting up
setup(  name="CatLearn", 
        version=VERSION,
        author="Andreas Vishart",
        author_email="<alyvi@dtu.dk>",
        url="https://github.com/avishart/CatLearn/tree/PhD",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy','scipy','ase'], 
        keywords=['python','gaussian process','machine learning','regression'],
        classifiers= [
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X"
        ]
)
