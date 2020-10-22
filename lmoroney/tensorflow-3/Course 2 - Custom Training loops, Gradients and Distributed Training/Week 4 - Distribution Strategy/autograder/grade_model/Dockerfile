FROM frolvlad/alpine-miniconda3:python3.7

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir /grader && \ 
    mkdir /grader/data

COPY data/ /grader/data/
COPY disable_warnings.py /grader/disable_warnings.py
COPY tools.py /grader/tools.py
COPY converter.py /grader/converter.py
COPY grader.py /grader/grader.py

RUN chmod a+rwx -R /grader/

WORKDIR /grader/

ENTRYPOINT ["python", "grader.py"]