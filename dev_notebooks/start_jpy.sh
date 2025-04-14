#!/bin/bash -I

# Start Jupyter Notebook on set port with set token
jupyter notebook --port=9832 --NotebookApp.token='harambe' --no-browser

# http://localhost:9832/tree?token=harambe