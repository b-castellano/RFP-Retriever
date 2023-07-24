# FROM python:3.9-slim

# # RUN apt-get update --fix-missing && apt-get install -y \
# #     build-essential \
# #     curl \
# #     software-properties-common \
# #     git \
# #     && rm -rf /var/lib/apt/lists/*

# # RUN git clone <some repo link here> .

# COPY requirements.txt requirements.txt
# RUN pip3 install --upgrade pip
# RUN apt-get update --fix-missing && apt-get upgrade -y && apt-get dist-upgrade -y
# RUN apt-get install gcc musl-dev -y
# RUN pip3 install -r requirements.txt

# WORKDIR /VectorDB-Searcher

# COPY . .

# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.9.6
WORKDIR /RFP-Retriever
COPY . .
RUN pip3 install -r requirements.txt
EXPOSE 8501
CMD python3 -m streamlit run app.py
