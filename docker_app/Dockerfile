FROM ubuntu:20.04

# Install Python and Jupyter Notebook
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install jupyter notebook

# Copy the notebook file to the container
COPY app_projekt_6.ipynb /notebooks/my_notebook.ipynb

# Set the working directory and start Jupyter Notebook
WORKDIR /notebooks
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
