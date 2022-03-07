from setuptools import setup, find_packages

# TODO -> Migrate to pyproject.toml and setup.cfg

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    reqs = f.read()

setup(
    name="ingeniator",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs.strip().split("\n"),
    author="Mike Lasby",
    description="TBD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=["Programming Language :: Python :: 3.9"],
    test_suite="pytest",
)
