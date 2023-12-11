FROM ghcr.io/dask/dask:2023.10.0-py3.10

COPY batch-jobs src requirements_clean.txt ./

RUN pip install -r requirements_clean.txt 

# Maven is for SUTime
RUN apt update && \
    python -m nltk.downloader averaged_perceptron_tagger punkt wordnet && \
    DEBIAN_FRONTEND='noninteractive' apt install -y maven
RUN mvn dependency:copy-dependencies -DoutputDirectory=./jars -f $(python3 -c 'import importlib; import pathlib; print(pathlib.Path(importlib.util.find_spec("sutime").origin).parent / "pom.xml")')

### GCS Authentification
# Next Line has to be handled somewhere else... secret!
# COPY extreme-lore-398917-ac46de419eb2.json ./
ENV GOOGLE_APPLICATION_CREDENTIALS=/extreme-lore-398917-ac46de419eb2.json
