from setuptools import setup, find_packages

# Read the contents of README.md for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="gra-physical-ai",
    version="0.1.0",
    author="qqewq",
    author_email="",  # add your email if desired
    description="Open platform for building aligned, safe, and ethical physical AI agents using multilevel GRA Meta‑Nullification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qqewq/gra-physical-ai",
    project_urls={
        "Bug Tracker": "https://github.com/qqewq/gra-physical-ai/issues",
        "Documentation": "https://github.com/qqewq/gra-physical-ai/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)