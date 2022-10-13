Setup Conda Environment
=====================================

This setup ensures that we are working in the same python environment 

# Creating an environment with commands

Use the terminal or an Anaconda Prompt for the following steps:

    conda create --name myenv

Replace ``myenv`` with the environment name.

    conda activate myenv

Activate the environment

# Creating an environment with our project packages


Navigate to this folder path

Replace ``path/to/this/folder`` with the correct path

    cd path/to/this/folder


Create an environment with packages using this command

    conda create --name CS124H --file req.txt

# Update ``req.txt`` when install new packages for project 


    conda activate CS124H
    conda list -e > req.txt
    
A new requirement file will be created in the folder and you can push it on github.

____________
List of packages
=====================================
- numpy=1.23.1=py310hdcd3fac_0
- numpy-base=1.23.1=py310hfd2de13_0
- pandas=1.4.4=py310he9d5cce_0
- pytorch=1.10.2=cpu_py310h30e64cd_0