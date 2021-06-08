# ai-tests


An environment for deep learning with tensorflow/keras and test scripts.

---

## Installation 

Install python 3.9.5 with miniconda:

on Linux:

	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	  bash Miniconda3-latest-Linux-x86.sh

install CUDA:
    
on MacOS:

	  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	  bash Miniconda3-latest-MacOSX-x86_64.sh
	  
	  
	  

go to a directory of your choice

	  git clone https://github.com/cmoestl/ai-tests


	  

Create a conda environment using the environment.yml in the repository root directory, 
for NVIDIA GPU usage, installation of CUDA is included.

	  conda env create -f environment.yml


This file looks like:

name: ait2
dependencies:
  - python==3.9.5
  - cudatoolkit 
  - pip
  - jupyterlab
  - numpy
  - numba
  - matplotlib
  - pandas
  - seaborn
  - tensorflow
  - keras
  - scikit-learn
  



Activate the environment:

	  conda activate ait2
   

Run scripts.

For Keras examples see:  https://github.com/keras-team/keras-io/tree/master/examples

   
---   

not necessary so far, add packages over pip 

	  pip install -r requirements.txt
	
    
	  
	  
Export dependencies:	  

    conda list --export > environment2.yml
    pip freeze > requirements2.txt
    
    
Links: https://anaconda.org/anaconda/cudatoolkit