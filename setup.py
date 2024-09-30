from setuptools import setup, find_packages

setup(
    name='J79X2',
    version='0.0.0',
    author='Nicholas Orndorff',
    author_email='norndorff@ucsd.edu',
    packages=find_packages(),
    python_requires='>=3.7',
    platforms=['any'],
    install_requires=[
        'numpy',
        'scipy',
        'gitpython',
        'setuptools',
    ],
    include_package_data=True,
    # package_data={
    #     'NRLMSIS2': ['data/altitude_0_1000_1.pkl'],
    #     'NRLMSIS2': ['data/density_0_1000_1.pkl'],
    #     'NRLMSIS2': ['data/temperature_0_1000_1.pkl'],
    # },
)
