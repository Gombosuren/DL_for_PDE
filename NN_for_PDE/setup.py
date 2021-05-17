#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:37:45 2021

@author: gombosurenatarbayr
"""


from setuptools import setup, find_packages;


setup(
        name = 'Deep learn for Partial Differential Equation ',
        version = '1.0',
        description = 'Python Distribution Utilities',
        author = 'Gombosuren Atarbayar, MUST',
        author_email = 'g0m60suren.4@gmail.com',
        url = 'https://github.com/Gombosuren/DL_for_PDE.git',
        packages = find_packages(),
        install_requires=[
            'numpy>1.18',
            'tensorflow > 2.1.0'
        ]
    );