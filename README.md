# ai-tests


---

## Installation 

Install python 3.9.5 with miniconda:

on Linux:

	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	  bash Miniconda3-latest-Linux-x86.sh

on MacOS:

	  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	  bash Miniconda3-latest-MacOSX-x86_64.sh

go to a directory of your choice

	  git clone https://github.com/cmoestl/ai-tests
	  

Create a conda environment using the environment.yml and requirements.txt file in the heliocats root directory, and activate the environment in between:

	  conda env create -f environment.yml

	  conda activate helio

	  pip install -r requirements.txt
	  
	  
Export dependencies:	  

    conda list --export > environment2.yml
    pip freeze > requirements2.txt