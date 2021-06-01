# A tool to help you label massive NLP datasets, quickly.

## Installation

*We highly recommend creating a virtual environment before installing any libraries*.

First clone the repo by shelling:

```git clone git@github.com:artefactory/bulk_labelling.git && cd bulk_labelling/```

Install the necessary packages by shelling:

```pip install -r requirements.txt```

At this point, there might be an installation error concerning the *gensim* library, which necessitates *Microsoft visual C++ 14.0+ to function. this can be downloaded and installed at the following link: https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/.

At this point, reattempt installation if you did not have MVC14.0 installed. otherwise, you're good to go!

## Installing sample datasets and local models

You don't have quite all the necessary elements yet. If you want to explore sample datasets and use some of the pretrained models included in the repo, you'll have to run an additional bash script by shelling :

```bash install.sh```
## Launching and labelling your first dataset

to launch the app, simply shell:

```streamlit run main_app.py```
