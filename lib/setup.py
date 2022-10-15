from setuptools import find_packages, setup
setup(
    name='ft545',
    packages=find_packages(include=['ft545']),
    version='0.2.0',
    description='My first Python library',
    author='Me',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    test_suite='tests',
)