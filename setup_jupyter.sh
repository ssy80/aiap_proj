#!/bin/bash 

#setup jupyter

sudo apt install jupyter-core

python3 -m venv venv

source venv/bin/activate 

pip install jupyter ipykernel

pip install pandas
pip install matplotlib
pip install seaborn
pip install sqlalchemy
pip install scipy
pip install statsmodels
pip install PyYAML
pip install scikit-learn
pip install xgboost
pip install pytest
pip install category_encoders
pip install numpy
pip install datetime