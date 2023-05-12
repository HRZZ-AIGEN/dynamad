from setuptools import find_packages, setup

setup(
    name='dynamad',
    packages=find_packages(include=['dynamad']),
    version='0.1.0',
    description='Dynamic applicability domain (dAD) '+\
                'for the interaction prediction problem, '+\
                'encompassing point predictions with prediction intervals '+\
                'and confidence estimates.',
    author='davoors',
    license='MIT',
    install_requires=['numpy', 'pandas', 'tqdm'],
    test_suite='tests'
)