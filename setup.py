import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

requirements = HERE / 'requirements.txt'

with requirements.open(mode='rt', encoding='utf-8') as fp:
    install_requires = [line.strip() for line in fp]

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setuptools.setup(
    name="HuClip The Text",
    version="0.0.1",
    description="Compares questions with images on Hungarian language.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Tamás Ficsor",
    author_email="ficsort@inf.u-szeged.hu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python"
    ],
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.9"
)