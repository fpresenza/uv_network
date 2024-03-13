# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uvnpy",
    version="0.0.1",
    license='GPLv3',
    author="J. Francisco Presenza",
    author_email='jpresenza@fi.uba.ar',
    description=u'Módulos python',
    long_description=long_description,
    long_description_content_type="text/rst",
    url='https://gitlab.com/fpresenza/uv_network',
    packages=setuptools.find_packages(),
    package_data={'uvnpy': ['config/*.yaml']},
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License v3  \
        or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)
