PROJECT TITLE: Avian Acoustic Recognition Applications
AUTHOR: Donaghy "Conlan" Henson

#####################

Program Instructions:
## Installation
    1. To set up the necessary environment, first clone the repository
       using the line below in a shell environment:
        `git clone <url-to-repo> <name-of-repo>`
    2. Next, navigate to the repository directory:
        `cd <name-of-repo>`
    3. Now, this is optional but highly recommended as to avoid system
       conflicts, create a local Python virtual environment:
        `python -m venv <venv-name>`
    4. After running the script for step 3, initialize the virtual
       environment with the following command:
        `source <venv-name>/bin/activate`
    5. Next, install the necessary package requirements which
       have been conveniently placed in a text file within the
       GitHub repo using the following command:
        `pip install -r requirements.txt`
    6. That's it!

## Running the Program
    1. In order to run the program, the prior steps outlined in the
       'Installation' section must be completed for the program
       to run properly.
    2. After completing the 'Installation' steps, the next step
       is to run the program using the following shell command:
        'python avian-acoustics.py'
    3. If you are having trouble with your program, the most
       likely area that is giving you trouble is the folder
       where your data is stored. For your program to run
       properly, please ensure the variable within the
       'avian-acoustics.py' python file declared 'data_dir'
       contains the correct file path to extract the audio
       files from.
    4. That's it!

#####################

## Program Description:
    The following set of files within this GitHub repository represents
    my personal work in developing an open-source application for
    detecting bird species from audio samples, i.e., Avian Acoustic
    Recognition. The overall detection scheme was constructed with machine
    learning algorithms (Support Vector Machines stacked with Gradient Boosters)
    that were trained from the given data samples. The magic of this
    particular avian acoustic recognition applicaiton is the verbose
    emphasis on pre-processing the data before inputting into the
    machine learning method. This ensured that all of the data was brought
    to a baseline level of 'cleanliness' (known as normalization) before
    inputting into the machine learning method, which does not function properly
    when given raw data samples. By normalizing the data, the algorithm
    can have consistent samples to process which makes the classification
    process more accurate. Libraries like Librosa and SciKit-Learn have
    'off-the-shelf' methods specifically designed for audio processing
    tasks which were used extensively throughout this project and contributed
    greatly to the overall performance of the machine learning model. If you have
    any questions or would like to contribute to the project, please contact
    me at: donaghy.henson@gmail.com
    Thanks for checking out my project and have a fun time recognizing bird sounds!

#####################