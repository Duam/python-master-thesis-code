from setuptools import setup, find_packages

setup(
    name='Master-thesis-code',
    version="1.0.0",
    setup_requires=['setuptools'],
    use_scm_version=False,
    install_requires=[
        'numpy',
        'casadi',
        'matplotlib',
        #  'zerocm',
        'seaborn',
        'scipy',
        'pandas',
        'tikzplotlib',
        'xarray',
        'pynput'
    ],
    python_requires='>=3.6',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    license="MIT",
    author="Paul Daum",
    author_email="paul.daum@posteo.de",
    description="Contains the python code used in my thesis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            #  'start = carousel_control_loop_physical.main',
        ]
    },
)