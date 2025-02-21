from setuptools import setup, find_packages

setup(
    name="cfi_amd",  # Replace with your package name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "lightning==2.4.0",
        "opencv-python-headless==4.10.0.84",
        "scipy==1.14.1",
        "scikit-learn==1.5.2",
        "pydicom==3.0.1",
        "pandas==2.2.3",
        "matplotlib==3.10.0",
        "scikit-image==0.24.0",
    ],
    author="Bart Liefers",
    author_email="b.liefers@erasmusmc.nl",
    description="Segmentation of intermediate AMD features on color fundus images",
    url="https://github.com/Eyened/cfi-amd",
)
