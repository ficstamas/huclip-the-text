import pathlib
import setuptools
from distutils.command.sdist import sdist as sdist_orig
from distutils.errors import DistutilsExecError

# The directory containing this file
HERE = pathlib.Path(__file__).parent


class sdist(sdist_orig):
    def run(self):
        try:
            self.spawn(['requirements_manual.sh', ])
        except DistutilsExecError:
            self.warn('Installing dependencies failed')
        super().run()


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
    cmdclass={
        'sdist': sdist
    },
    python_requires=">=3.9"
)