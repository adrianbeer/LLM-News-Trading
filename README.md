# PROJECT B U G A T T I

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
