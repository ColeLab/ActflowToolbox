#! /usr/bin/env python

DISTNAME = "actflow"
DESCRIPTION = None  # TODO
MAINTAINER = None  # TODO
MAINTAINER_EMAIL = None  # TODO
LICENSE = None  # TODO
URL = 'https://colelab.github.io/ActflowToolbox/'
DOWNLOAD_URL = 'https://github.com/ColeLab/ActflowToolbox'
VERSION = "0.2.3.dev0"
PYTHON_REQUIRES = ">=3.6"
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'nibabel',
]
PACKAGES = [
    'actflow',
    'actflow.actflowcomp',
    'actflow.connectivity_estimation',
    'actflow.model_compare',
    'actflow.simulations',
    'actflow.tools',
]
SCRIPTS = [
]
CLASSIFIERS = [  # TODO
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
]
INCLUDE_PACKAGE_DATA = True


if __name__ == '__main__':

    from setuptools import setup

    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        url=URL,
        download_url=DOWNLOAD_URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        scripts=SCRIPTS,
        classifiers=CLASSIFIERS,
        include_package_data=INCLUDE_PACKAGE_DATA,
    )
