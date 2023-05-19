# ActiveStateAnalyser
ActiveStateAnalyser software package allows to measure statistics describing neuronal network state. It has clear interface, visible workflow and obtained results could be extracted for next statistical processing.

## Quick Start Guide
1. Download and unzip the code folder 
2. Install [Anaconda](https://www.anaconda.com/)
3. Open Anaconda 
4. Execute
```cmd
cd PATH_TO_CODE
conda create --name neuron-analysis -c conda -forge --file requirements.txt –y
```

## Run
1. Open Anaconda
2. Execute
```cmd
cd PATH_TO_CODE
conda activate neuron-analysis
jupyter notebook
```
After this steps, the joined Jupiter Notebook, containing separated notebooks, will be open:
* ActiveStateAnalyzer
* Distance analysis
* Statistics&Shuffling
* MultipleShuffling
* Dimensionality reduction

A detailed guide to these notebooks is [here](docs/tutorial.pdf).