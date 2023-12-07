# PROJECT B U G A T T I

# Location and Backups of Data:
Onedrive:
Google Cloud Storage (GCS):
USB-Stick (v/ describe):

# 


# remove below to docker file or somewhere else ... not appropriate here
## Installation

!sudo apt update
!sudo apt install maven;

!pip install html2text
!pip install datefinder
!pip install -U dask[complete]
!pip install nltk
!pip install dateparser
!pip install pyngrok
!pip install sutime;

# This is required for sutime
!mvn dependency:copy-dependencies -DoutputDirectory=./jars -f $(python3 -c 'import importlib; import pathlib; print(pathlib.Path(importlib.util.find_spec("sutime").origin).parent / "pom.xml")');
