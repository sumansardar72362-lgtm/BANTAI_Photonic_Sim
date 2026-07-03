from setuptools import setup, find_packages

setup(
    name="kecia",
    version="1.0.0",
    author="S Sardar",
    description="Core SDK for Kecia Diamond-Glass Hybrid Photonic Chip",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
)