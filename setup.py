from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='CancerDeepLearning',
    url='CancerDeepLearning',
    author='https://github.com/BiomedicalMachineLearning/CancerDeepLearning.git',
    author_email='quan.nguyen@uq.edu.au',
    # package modules
    packages=['SimpleModel'],
    # dependencies
    install_requires=['numpy'],
    # version
    version='0.1',
    # license
    license='MIT',
    description='Developing a package using genomics and image data for cancer diagnosis',
)

