FROM ghcr.io/dask/dask:2023.10.0-py3.10

COPY requirements_clean requirements_google setup.py .
ADD src .

RUN pip install -r requirements_clean.txt && \
    pip install -r requirements_google.txt

RUN apt update && \
    python -m nltk.downloader averaged_perceptron_tagger punkt wordnet && \
    DEBIAN_FRONTEND='noninteractive' apt install -y maven

RUN mvn dependency:copy-dependencies -DoutputDirectory=./jars -f $(python3 -c 'import importlib; import pathlib; print(pathlib.Path(importlib.util.find_spec("sutime").origin).parent / "pom.xml")')

RUN pip install .
