# Optimisation of Deep Neural Networks using a Genetic Algorithm: A Comparative Study

## About
Implementation of the paper [_"Optimisation of Deep Neural Networks using a Genetic Algorithm: A Comparative Study"_](paper.pdf) by Tiago Gonçalves, Leonardo Capozzi, Ana Rebelo and Jaime S. Cardoso.

## Abstract
Deep learning algorithms have been challenging human performance in several tasks. Currently, most of the methods to design the architectures of these models and to select the training hyper-parameters are still based on trial-and-error strategies. However, practitioners recognise that there is a need for tools and frameworks that can achieve high-performing models almost automatically. We addressed this challenge using a meta-heuristics approach. We implemented a genetic algorithm with a variable length chromosome with three different benchmark data sets. In this comparative study, we vary the architectures of the convolutional and the fully- connected layers and the learning rate. The best models achieve accuracy values of 98.73%, 90.81% and 54.71% on MNIST, Fashion-MNIST and CIFAR-10, respectively.

## Clone this repository
To clone this repository, open a Terminal window and type:
```bash
$ git clone git@github.com:TiagoFilipeSousaGoncalves/optimizing-dl-parameters-pdeec.git
```
Then go to the repository's main directory:
```bash
$ cd optimizing-dl-parameters-pdeec
```

## Dependencies
### Install the necessary Python packages
We advise you to create a virtual Python environment first (Python 3.7). To install the necessary Python packages run:
```bash
$ pip install -r requirements.txt
```

## Data
To know more about the data used in this paper, please send an e-mail to  [**tiago.f.goncalves@inesctec.pt**](mailto:tiago.f.goncalves@inesctec.pt) or to [**leonardo.g.capozzi@inesctec.pt**](mailto:leonardo.g.capozzi@inesctec.pt).


## Usage
### Reproduce the Experiments
To reproduce the experiments:
```bash
$ python code/run_genetic_algorithm.py
```


### Plot the Results
To plot the results:
```bash
$ python code/run_generate_plots.py
```




## Citation
If you use this repository in your research work, please cite this paper:
```bibtex
@inproceedings{goncalvescapozzi2023recpad,
	author = {Tiago Gonçalves, Leonardo Capozzi, Ana Rebelo and Jaime S. Cardoso},
	title = {{Optimisation of Deep Neural Networks using a Genetic Algorithm: A Comparative Study}},
	booktitle = {29th Portuguese Conference in Pattern Recognition (RECPAD)},
	year = {2023},
    address = {Coimbra, Portugal}
}
```
