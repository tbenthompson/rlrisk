from setuptools import setup

version = open("VERSION").read()

description = open("README.md").read()

setup(
    packages=["rlrisk"],
    install_requires=[],
    zip_safe=False,
    name="rlrisk",
    version=version,
    description="",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/tbenthompson/rlrisk",
    author="T. Ben Thompson",
    author_email="t.ben.thompson@gmail.com",
    platforms=["any"],
    classifiers=[],
)
