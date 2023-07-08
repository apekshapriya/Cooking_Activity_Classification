from pathlib import Path
from setuptools import setup, find_packages

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent

file1 = open(Path(BASE_DIR, "requirements.txt"), "r")
lines = file1.readlines()


EGG_MARK = "#egg="
required_packages = [ln.strip() for ln in lines]

git_pack = [p for p in required_packages if p.startswith("-e git")]

for pack in git_pack:
    required_packages.remove(pack)

for pack in git_pack:
    pack = pack.strip("-e")

    if EGG_MARK in pack:
        pack_name = pack[pack.find(EGG_MARK) + len(EGG_MARK) :]
        repo = pack[: pack.find(EGG_MARK)]
        required_packages.append("%s @ %s" % (pack_name, repo))


# setup.py
setup(
    name="activity_classifier",
    packages=find_packages(include=["activity_classifier"]),
    version=0.1,
    description="Classify cooking activities",
    author="Apeksha Priya",
    author_email="apekshapriya@gmail.com",
    python_requires=">=3.8.10",
    install_requires=[required_packages],
)
