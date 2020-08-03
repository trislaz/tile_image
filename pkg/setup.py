import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiler_wsi-trela", # Replace with your own username
    version="0.0.1",
    author="Trite Zard",
    author_email="tristan.lazard@mines-paristech.fr",
    description="Multiple instance learning for wsi classification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trislaz/tile_image",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
