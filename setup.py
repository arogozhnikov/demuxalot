from setuptools import setup

setup(
    name="demultiplexit",
    version='0.1.0',
    description="Reliable demultiplexing for single-cell RNA sequencing that refines genotypes",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Alex Rogozhnikov',
    packages=['demultiplexit'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 ',
    ],
    keywords='genotype learning, single cell RNA sequencing, demultiplexing, bayesian modelling',
    install_requires=[
        'pysam',
        'scipy',
        'numpy',
        'joblib',
    ],
)