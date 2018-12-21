FROM python:2.7.12

RUN pip install --upgrade pip
RUN pip uninstall numpy
RUN pip uninstall matplotlib
RUN pip uninstall pandas
RUN pip uninstall seaborn
RUN pip uninstall scipy
RUN pip uninstall statsmodels

WORKDIR /home

# Generating the folders and files needed and populating them
RUN mkdir notebooks

COPY ./requirements* /home
COPY ./challenge-data-v2.csv /home
COPY ./notebooks /home/notebooks

# Installing the python packages
RUN pip install -r requirements.txt --no-cache-dir

# Making the port available locally
EXPOSE 8888

CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
