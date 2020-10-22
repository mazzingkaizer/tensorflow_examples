FROM frolvlad/alpine-miniconda3:python3.7

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir /grader && \ 
    mkdir /grader/solution && \
    mkdir /grader/submission && \
    mkdir /grader/data

COPY entry.py /grader/entry.py
COPY grader.py /grader/grader.py
COPY disable_warnings.py /grader/disable_warnings.py
COPY converter.py /grader/converter.py
COPY nb_parser.py /grader/nb_parser.py
COPY tools.py /grader/tools.py
COPY compiled.py /grader/compiled.py
COPY solution/* /grader/solution/

COPY sample_dataset_image.pkl /grader/
COPY sample_image.pkl /grader/
COPY data/ /grader/data/

RUN chmod a+rwx -R /grader/

WORKDIR /grader/

ENTRYPOINT ["python", "entry.py"]