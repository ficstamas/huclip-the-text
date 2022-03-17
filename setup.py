import pathlib
import setuptools
from distutils.command.sdist import sdist as sdist_orig
from distutils.errors import DistutilsExecError

# The directory containing this file
HERE = pathlib.Path(__file__).parent


deps_ = [
    'torch==1.11.0+cu113 -f "https://download.pytorch.org/whl/cu113/torch_stable.html"',
    'torchvision==0.12.0+cu113 -f "https://download.pytorch.org/whl/cu113/torch_stable.html"',
    'torchaudio===0.11.0+cu113 -f "https://download.pytorch.org/whl/cu113/torch_stable.html"'
    'ftfy',
    'transformers',
    'huspacy',
    'beautifulsoup4',
    'git+https://github.com/openai/CLIP.git',
    'git+https://github.com/ficstamas/paddle.git',
    'multi_rake'
]


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
    python_requires=">=3.9",
    install_requires=deps_
)