from setuptools import setup, find_packages

setup(
    name='preprocess_tools',
    version='0.1.9',
    author='Alberto Vicente del Egido',
    author_email='alberto.vicente@imdea.org',
    description='Database utilities and preprocessing tools for UT vs XCT database',
    long_description='Tools for preprocessing and managing data for the UT vs XCT project.',
    long_description_content_type='text/markdown',
    url='https://github.com/topeberti/UTvsXCT-preprocessing',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "joblib",
        "tqdm",
        "tifffile",
        "PyWavelets "
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Database',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    license='MIT',
    keywords='database preprocessing UT XCT',
    project_urls={
        'Source': 'https://github.com/topeberti/UTvsXCT-preprocessing',
        'Tracker': 'https://github.com/topeberti/UTvsXCT-preprocessing/issues',
    },
)
