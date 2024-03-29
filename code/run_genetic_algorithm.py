# Imports
import os
import numpy as np
import _pickle as cPickle
import random

# Torch Imports
import torch

# Project Imports
from code.genetic_algorithm import GeneticAlgorithm



# Set random seed value so we a have a reproductible work
random_seed = 42



# Initialize random seeds
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)



# Datasets and Datasets-Shapes
datasets_names = ["mnist", "fashion-mnist", "cifar10"]
datasets_shapes = [[1, 28, 28], [1, 28, 28], [3, 32, 32]]

# Only MNIST
# datasets_names = ["mnist"]
# datasets_shapes = [[1, 28, 28]]

# Only Fashion-MNIST
# datasets_names = ["fashion-mnist"]
# datasets_shapes = [[1, 28, 28]]

# Only CIFAR-10
# datasets_names = ["cifar10"]
# datasets_shapes = [[3, 32, 32]]

# We define this function to avoid issues with Windows OS
def main():
    # Go through datasets and shapes
    for dataset, shape in zip(datasets_names, datasets_shapes):
        # Some debug prints
        print(f"Current Dataset {dataset} | Dataset Shape: {shape}")

        if not os.path.isdir(f"results/{dataset}"):
            os.makedirs(f"results/{dataset}")

        # Create GeneticAlgorithm Instance
        ga = GeneticAlgorithm(input_shape=shape, size_of_population=40, nr_of_labels=10,
                              nr_of_phases=5, nr_of_generations=100, nr_of_autoselected_solutions=4,
                              mutation_rate=0.2, initial_chromossome_length=2, nr_of_epochs=5, data=dataset)

        # Train the genetic algorithm
        ga.train()

        # Test the genetic algorithm
        results = ga.test(epochs=30)

        # Save results into a pickle file for further analysis
        with open(f"results/{dataset}/test_results.pickle", 'wb') as fp:
            cPickle.dump(results, fp, -1)

    print(f"Finished the training of all datasets")


# Run the algorithm
if __name__ == '__main__':
    main()