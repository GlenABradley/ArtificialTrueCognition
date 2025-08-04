from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="syncopation_engine",
    version="0.1.0",
    author="Glen Bradley",
    author_email="glen@example.com",
    description="Syncopation Engine for Artificial True Cognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GlenABradley/ArtificialTrueCognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.20.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scikit-learn>=0.24.0',
        'hdbscan>=0.8.27',
        'opencv-python>=4.5.0',
    ],
    include_package_data=True,
)
