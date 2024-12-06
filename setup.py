from setuptools import find_packages, setup

setup(
    name="inference_rayservice",
    version="0.1.0",
    description="Code for the Elevaite Inference Ray Service",
    author="Connor Boyle, Korin Thompson",
    packages=find_packages(),
    install_requires=[
        "ray[serve]>=2.37.0",
        "transformers>=4.45.1",
        "sentence-transformers>=2.7.0",
        "torch>=2.4.1",
        "fastapi>=0.115.0",
        "numpy>=2.1.1",
    ],
    python_requires=">=3.11.9",
)
