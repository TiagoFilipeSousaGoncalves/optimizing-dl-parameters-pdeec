# Imports
import numpy as np
import time
import random
import os

# Torch Imports
import torch
import torch.nn as nn
from torchsummary import summary

# Project Imports
from code.model import Model
from code.utilities import utils
from code.datasets import get_mnist_loader, get_fashion_mnist_loader, get_cifar10_loader

# Sklearn Imports
import sklearn.metrics as sklearn_metrics

# Pickle
import _pickle as cPickle

# Set random seed value so we a have a reproductible work
random_seed = 42

# Initialize random seeds
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


# Define the Genetic Algorithm Class
class GeneticAlgorithm:
    def __init__(self, input_shape, size_of_population, nr_of_labels, nr_of_phases, nr_of_generations, nr_of_autoselected_solutions,
                 mutation_rate, initial_chromossome_length, nr_of_epochs=5, data="mnist"):
        
        # Assert that the size of population must be even
        assert size_of_population % 2 == 0, "Size of population must be even."

        # Dataset Variables
        self.input_shape = input_shape
        self.nr_of_labels = nr_of_labels

        # Initialize variables
        self.size_of_population = size_of_population
        self.nr_of_generations = nr_of_generations
        self.mutation_rate = mutation_rate
        self.nr_of_autoselected_solutions = nr_of_autoselected_solutions
        self.nr_of_epochs = nr_of_epochs
        self.data_name = data

        # Phase Variables
        self.nr_of_phases = nr_of_phases

        # Chromossome Length Variables
        self.current_chromossome_length = initial_chromossome_length

        self.best_solution = list()
        self.best_model = None

        self.best_solution_aux = list()
        self.best_model_aux = None

        self.best_sol_fitness_aux = -np.Inf
        self.best_sol_fitness = -np.Inf


    # Function: Copy Solution Object
    def copy_solution(self, sol):
        return [sol[0].clone(), sol[1].clone(), sol[2].clone()]

    # Function: Generate Random Solution
    def generate_random_solution(self, chromossome_length, input_shape):
        # Block Sizes
        nr_conv_layers = np.random.randint(0, chromossome_length + 1)
        nr_fc_layers = chromossome_length - nr_conv_layers

        # Process Input Shape [C, H, W]
        channels = input_shape[0]
        rows = input_shape[1]
        columns = input_shape[2]

        # Create Convolutional Block
        convolutional_layers = []
        curr_tensor = torch.rand((1, channels, rows, columns))

        for _ in range(nr_conv_layers):
            curr_layer = []

            # Column 0: Number of Conv-Filters
            nr_filters = random.choice(utils.conv_nr_filters)
            curr_layer.append(nr_filters)

            # Column 1: Conv-Kernel Sizes
            max_kernel_size = min(curr_tensor.size()[2:])
            allowed_conv_kernel_size = utils.conv_kernel_size[utils.conv_kernel_size <= max_kernel_size]
            kernel_size = random.choice(allowed_conv_kernel_size)
            curr_layer.append(kernel_size)

            # Update curr_tensor
            curr_tensor = nn.Conv2d(in_channels=curr_tensor.size()[1], out_channels=nr_filters, kernel_size=kernel_size)(curr_tensor)

            # Column 2: Conv-Activation Functions
            activ_function = random.randint(0, len(utils.conv_activ_functions)-1)
            curr_layer.append(activ_function)

            # Column 3: Conv-Drop Rates
            drop_out = random.uniform(utils.conv_drop_out_range[0], utils.conv_drop_out_range[1])
            curr_layer.append(drop_out)

            # Column 4: Conv-Pool Layer Types
            max_kernel_size = min(curr_tensor.size()[2:])

            if max_kernel_size < 2:
                pool = 0
            else:
                pool = random.randint(0, len(utils.conv_pooling_types)-1)

            # Update curr_tensor
            curr_tensor = utils.conv_pooling_types[pool](curr_tensor)

            curr_layer.append(pool)

            # Add to convolutional block
            convolutional_layers.append(curr_layer)

        # Create Fully Connected Block
        fully_connected_layers = []

        for _ in range(nr_fc_layers):
            curr_layer = []

            # Column 0: FC Layer Number of Neurons
            nr_neurons = random.randint(utils.fc_nr_neurons_range[0], utils.fc_nr_neurons_range[1])
            curr_layer.append(nr_neurons)

            # Column 1: FC Layer Activation Functions
            activ_function = random.randint(0, len(utils.fc_activ_functions) - 1)
            curr_layer.append(activ_function)

            # Column 2: FC Dropout Rates
            drop_out = random.uniform(utils.fc_drop_out_range[0], utils.fc_drop_out_range[1])
            curr_layer.append(drop_out)

            fully_connected_layers.append(curr_layer)

        # Learning Rate
        learning_rate = random.choice(utils.learning_rate)

        convolutional_layers = torch.tensor(convolutional_layers)
        fully_connected_layers = torch.tensor(fully_connected_layers)
        learning_rate = torch.tensor([learning_rate])

        return [convolutional_layers, fully_connected_layers, learning_rate]


    # Function: Generate a List of Candidate Solutions
    def generate_candidate_solutions(self):
        # Create list to append solutions
        list_of_candidate_solutions = []

        # Go through the size of the population
        # Initialize current p
        p = 0
        # We have to build p == size_of_population solutions
        while p < self.size_of_population:

            # Append this solution to the list of candidate solutions
            list_of_candidate_solutions.append(self.generate_random_solution(self.current_chromossome_length, self.input_shape))
            # Update current p
            p += 1

        return list_of_candidate_solutions

    # Function: Training Method
    def train(self):
        # Load data
        # Data will always be the same, so we can read it in the beginning of the loop
        # Choose data loader based on the "self.data" variable
        if self.data_name.lower() == "mnist":
            data_loader = get_mnist_loader(32)

        elif self.data_name.lower() == "fashion-mnist":
            data_loader = get_fashion_mnist_loader(32)

        elif self.data_name.lower() == "cifar10":
            data_loader = get_cifar10_loader(32)

        else:
            raise ValueError(f"{self.data_name} is not a valid argument. Please choose one of these: 'mnist', 'fashion-mnist', 'cifar10'.")

        stat_data = np.zeros((self.nr_of_phases, self.nr_of_generations, self.size_of_population, 4), dtype=object)

        current_phase = 0

        # Evaluate the current phase against the maximum number of phases
        while current_phase < self.nr_of_phases:
            print(f"Current Training Phase: {current_phase}")

            # In each phase we begin with current_generation == 0
            current_generation = 0

            # Evaluate current generation against the maximum number of generations
            while current_generation < self.nr_of_generations:
                # Check if we are in current_generation == 0
                if current_generation == 0:
                    # Generate initial candidate solutions
                    # (this list will be usefull to the next steps such as mutation and crossover)
                    gen_candidate_solutions = self.generate_candidate_solutions()
                    print(f"Generation {current_generation} solutions generated.")

                # If not, we are generating not new solutions; we obtain new ones by crossover and mutation
                else:
                    # Apply crossover between best solutions of the previous generation until you achieve
                    # the size of the population
                    best_gen_candidates = list()

                    # Append the previous best nr of autoselected solutions to assure that progress is not lost
                    # Get the sorted indices 
                    sorted_indices = np.argsort(generation_solutions_fitness)
                    # Use this var self.nr_of_autoselected_solutions
                    for i in range(-1, -1-self.nr_of_autoselected_solutions, -1):
                        # The n best solutions are in the end
                        best_gen_candidates.append(gen_candidate_solutions[sorted_indices[i]])

                    print(f"Number of Best Candidates Selected:{len(best_gen_candidates)}")

                    # Apply Most Fit Solutions Methods
                    print(f"Selecting best solutions...")
                    most_fit_solutions = self.solution_selection(s_population=gen_candidate_solutions, s_fitnesses=generation_solutions_fitness)

                    # Shuffle most fit solutions
                    np.random.shuffle(most_fit_solutions)

                    # Reset gen_candidate_solutions
                    gen_candidate_solutions = list()

                    # Iterate through most fit solutions
                    # We need to subtract self.nr_of_autoselected_solutions to keep the size of population!
                    for idx in range(0, len(most_fit_solutions)-self.nr_of_autoselected_solutions, 2):
                        sol1, sol2, = self.apply_crossover(sol1=most_fit_solutions[idx], sol2=most_fit_solutions[idx+1])
                        gen_candidate_solutions.append(sol1)
                        gen_candidate_solutions.append(sol2)

                    print(f"Generation {current_generation} solutions' crossover applied.")

                    # Apply random mutations to the population
                    gen_candidate_solutions = self.apply_mutation(alive_solutions_list=gen_candidate_solutions)
                    print(f"Generation {current_generation} solutions' mutations applied.")

                    # Repair solutions so we have a complete list of feasible solutions
                    gen_candidate_solutions = [self.repair_solution(s) for s in gen_candidate_solutions]
                    print(f"Generation {current_generation} solutions generated and repaired.")

                    # Concatenate gen_candidate_solutions and best_gen_candidates
                    gen_candidate_solutions = best_gen_candidates + gen_candidate_solutions

                # Create models list
                models = []

                # Create models from these candidate solution
                # If we are in the 1st phase, we only need to generate new models
                if current_phase == 0:
                    for candidate in gen_candidate_solutions:
                        models.append(Model(self.input_shape, self.nr_of_labels, candidate))

                # If we are in an advanced phase, we have to perform transfer learning to assure fairness
                # in solutions with longer cromossomes lengths
                else:
                    for candidate in gen_candidate_solutions:
                        models.append(self.transfer_learning(previous_model=self.best_model,
                                                             previous_best_solution=self.best_solution,
                                                             new_candidate_solution=candidate))

                # Choose loss function; here we use CrossEntropy
                loss = nn.CrossEntropyLoss()

                # Select device: GPU or CPU
                device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

                # Total Time Generation (Start)
                generation_start = time.time()

                # Models Results (Generation)
                generation_models_results = list()

                # Training Loop
                for model_idx, model in enumerate(models):
                    print(f'Training Model {model_idx+1}')

                    # Transfer model to device (CPU or GPU)
                    model = model.to(device)

                    # Put model in "training mode"
                    model.train()

                    # Load optimizer (for now, we will used Adam)
                    optim = torch.optim.Adam(model.parameters(), lr=model.learning_rate)

                    # Model Starting Time
                    model_start = time.time()

                    # We train each model for 5 Epochs
                    for epoch in range(self.nr_of_epochs):

                        # Epoch Start Time
                        epoch_start = time.time()

                        # Running loss is initialised as 0
                        running_loss = 0.0

                        # Initialise y and y_pred lists to compute the accuracy in the end of the epoch
                        y = list()
                        y_pred = list()

                        # Iterate through the dataset
                        for i, data in enumerate(data_loader):

                            images, labels = data

                            # zero the parameter gradients
                            optim.zero_grad()

                            # forward + backward + optimize
                            images = images.to(device)
                            labels = labels.to(device)

                            features = model(images)

                            loss_value = loss(features, labels)

                            loss_value.backward()

                            optim.step()

                            # Get statistics
                            running_loss += loss_value.item() * images.size(0)

                            # Concatenate lists
                            y += list(labels.cpu().detach().numpy())
                            y_pred += list(torch.argmax(features, dim=1).cpu().detach().numpy())

                        # Average Train Loss
                        avg_train_loss = running_loss/len(data_loader.dataset)

                        # Train Accuracy
                        train_acc = sklearn_metrics.accuracy_score(y_true=y, y_pred=y_pred)

                        # Epoch End Time
                        epoch_end = time.time()

                        print(f"Epoch: {epoch+1} | Time: {epoch_end-epoch_start} | Accuracy: {train_acc} | Loss: {avg_train_loss}")

                        # torch.save(model.state_dict(), 'model.pth')

                    # Model Ending Time 
                    model_end = time.time()
                    total_model_time = model_end - model_start
                    print(f'Finished Training after {total_model_time} seconds.')

                    # Pass model to the CPU 
                    model = model.cpu()

                    # Append models results to the generation models results list to evaluate fitness
                    # We append: [train_acc, avg_train_loss, total_model_time]
                    generation_models_results.append([train_acc, avg_train_loss, total_model_time])

                # Generation Total Time
                generation_end = time.time()
                generation_total_time = generation_end - generation_start
                print(f"Finished Generation {current_generation} | Total Time: {generation_total_time}")

                # Evaluate Generations Solutions Fitness
                # Obtain fitness values
                generation_solutions_fitness = [self.solution_fitness(r[0]) for r in generation_models_results]
                # print(generation_solutions_fitness)

                # Save Solution and Save Statistic Data
                for stat_idx in range(len(generation_models_results)):
                    stat_data[current_phase][current_generation][stat_idx][0] = generation_models_results[stat_idx][0]
                    stat_data[current_phase][current_generation][stat_idx][1] = generation_models_results[stat_idx][1]
                    stat_data[current_phase][current_generation][stat_idx][2] = generation_solutions_fitness[stat_idx]
                    stat_data[current_phase][current_generation][stat_idx][3] = gen_candidate_solutions[stat_idx]
                # print(stat_data)

                # Save statistics into a NumPy Array
                stat_path = f"results/{self.data_name.lower()}"
                stat_filename = "stat_data.pickle"

                # Create directories if they are not created yet
                if not os.path.isdir(stat_path):
                    os.makedirs(stat_path)

                with open(os.path.join(stat_path, stat_filename), 'wb') as fp:
                    cPickle.dump(stat_data, fp, -1)

                # np.save(file=os.path.join(stat_path, stat_filename), arr=stat_data, allow_pickle=True)

                best_model_idx = np.argmax(generation_solutions_fitness)

                # Update best model, best solution and best solution fitness variables
                if generation_solutions_fitness[best_model_idx] > self.best_sol_fitness_aux:
                    self.best_model_aux = models[best_model_idx]
                    self.best_solution_aux = gen_candidate_solutions[best_model_idx]
                    self.best_sol_fitness_aux = generation_solutions_fitness[best_model_idx]

                current_generation += 1


            # Check if the previous phase helped
            # If it helped, continue
            if self.best_sol_fitness_aux > self.best_sol_fitness:
                self.best_model = self.best_model_aux
                self.best_solution = self.best_solution_aux
                self.best_sol_fitness = self.best_sol_fitness_aux

                # Update phase
                current_phase += 1

                # Update chromossome length
                self.current_chromossome_length += 1
            
            # Else, proceed to the testing phase
            else:
                current_phase = self.nr_of_phases

    # Function: Test Phase
    def test(self, epochs=30):
        # Load data
        # Data will always be the same, so we can read it in the beginning of the loop
        # Choose data loader based on the "self.data" variable
        # Here we load the data 
        if self.data_name.lower() == "mnist":
            train_loader = get_mnist_loader(batch_size=32, train=True)
            test_loader = get_mnist_loader(batch_size=32, train=False)
            
        
        elif self.data_name.lower() == "fashion-mnist":
            train_loader = get_fashion_mnist_loader(batch_size=32, train=True)
            test_loader = get_fashion_mnist_loader(batch_size=32, train=False)
        
        elif self.data_name.lower() == "cifar10":
            train_loader = get_cifar10_loader(batch_size=32, train=True)
            test_loader = get_cifar10_loader(batch_size=32, train=False)
        
        else:
            raise ValueError(f"{self.data_name} is not a valid argument. Please choose one of these: 'mnist', 'fashion-mnist', 'cifar10'.")

        
        # Choose loss function; here we use CrossEntropy
        loss = nn.CrossEntropyLoss()

        # Select device: GPU or CPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        # Load the best model
        # model = Model(self.input_shape, self.number_of_labels, self.best_solution)

        print(f'Training Best Model on Train Set')

        # Transfer model to device (CPU or GPU)
        self.best_model = self.best_model.to(device)

        # Put model in "training mode"
        self.best_model.train()

        # Load optimizer (for now, we will used Adam)
        optim = torch.optim.Adam(self.best_model.parameters(), lr=self.best_model.learning_rate)

        # Model Starting Time
        model_start = time.time()

        train_loss_min = np.Inf

        # We train each model for 30 Epochs
        for epoch in range(epochs):
            
            # Epoch Start Time
            epoch_start = time.time()
            
            # Running loss is initialised as 0
            running_loss = 0.0

            # Initialise y and y_pred lists to compute the accuracy in the end of the epoch
            y = list()
            y_pred = list()

            # Iterate through the dataset
            for i, data in enumerate(train_loader):

                images, labels = data

                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                images = images.to(device)
                labels = labels.to(device)

                features = self.best_model(images)

                loss_value = loss(features, labels)

                loss_value.backward()

                optim.step()

                # Get statistics
                running_loss += loss_value.item() * images.size(0)

                # Concatenate lists
                y += list(labels.cpu().detach().numpy())
                y_pred += list(torch.argmax(features, dim=1).cpu().detach().numpy())

            # Average Train Loss
            avg_train_loss = running_loss/len(train_loader.dataset)

            # Train Accuracy
            train_acc = sklearn_metrics.accuracy_score(y_true=y, y_pred=y_pred)

            # Epoch End Time
            epoch_end = time.time()

            print(f"Epoch: {epoch+1} | Time: {epoch_end-epoch_start} | Accuracy: {train_acc} | Loss: {avg_train_loss}")

            if avg_train_loss < train_loss_min:
                print(f"Train Loss decreased from {train_loss_min} to {avg_train_loss}.")

                model_path = f'results/{self.data_name.lower()}'
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)

                torch.save(self.best_model.state_dict(), os.path.join(model_path, 'best_model_weights.pt'))
                train_loss_min = avg_train_loss

        
        # Testing Phase
        print(f"Testing Best Model on Test Set")

        self.best_model = self.best_model.eval()


        with torch.no_grad():
            # Running loss is initialised as 0
            running_loss = 0.0

            # Initialise y and y_pred lists to compute the accuracy in the end of the epoch
            y = list()
            y_pred = list()

            # Iterate through the dataset
            for i, data in enumerate(test_loader):

                images, labels = data

                # forward + backward + optimize
                images = images.to(device)
                labels = labels.to(device)

                features = self.best_model(images)

                loss_value = loss(features, labels)

                # Get statistics
                running_loss += loss_value.item() * images.size(0)

                # Concatenate lists
                y += list(labels.cpu().detach().numpy())
                y_pred += list(torch.argmax(features, dim=1).cpu().detach().numpy())

            # Average Train Loss
            avg_test_loss = running_loss/len(train_loader.dataset)

            # Train Accuracy
            test_acc = sklearn_metrics.accuracy_score(y_true=y, y_pred=y_pred)

        print(f"Accuracy: {test_acc} | Loss: {avg_test_loss}")

        return [test_acc, avg_test_loss]


    # TODO: Implement Thread Training Solution
    def thread_training(self):
        pass

    # Transfer Learning Function
    # TODO: debug: transfer weights from a pretrained network to a new one and check the accuracy
    def transfer_learning(self, previous_model, previous_best_solution, new_candidate_solution):
        # Put in train mode
        previous_model.train()

        # Create The New Model Instance 
        pretrained_model = Model(self.input_shape, self.nr_of_labels, new_candidate_solution)
        pretrained_model.train()

        # Create a list with the two models: it will be useful with the next steps
        # models_list = [previous_model, pretrained_model]

        # Check conv-layers first
        # Create a list with the conv-blocks of each solution
        conv_layers_sols = [previous_best_solution[0], new_candidate_solution[0]]

        # Check the solution which has the lower number of layers
        nr_of_conv_layers = [np.shape(conv_layers_sols[0])[0], np.shape(conv_layers_sols[1])[0]]
        limitant_sol_idx = np.argmin(nr_of_conv_layers)

        # Transfer Weights
        with torch.no_grad():
            conv_idx = 0
            curr_idx = 0
            # We iterate through the number of layers equal to the limitant_sol_idx
            while conv_idx < nr_of_conv_layers[limitant_sol_idx]:
                if isinstance(previous_model.convolutional_layers[curr_idx], nn.Conv2d):
                    # Access this index and see the size of weight and bias tensors
                    # Weight Tensors
                    # Previous Weights
                    previous_weights = torch.clone(previous_model.convolutional_layers[curr_idx].weight)
                    previous_weights_size = previous_model.convolutional_layers[curr_idx].weight.size()
                    # New Weights
                    new_weights = torch.clone(pretrained_model.convolutional_layers[curr_idx].weight)
                    new_weights_size = pretrained_model.convolutional_layers[curr_idx].weight.size()


                    # Each Conv2d Tensor has 4 Dimensions; we need to get maximum dimension size
                    # Dimension 0: The Output Channels
                    max_nr_out_channels = max(int(previous_weights_size[0]), int(new_weights_size[0]))
                    # Dimension 1: The Input Channels
                    max_nr_in_channels = max(int(previous_weights_size[1]), int(new_weights_size[1]))
                    # Dimension 2: Kernel Size XX-Axis
                    max_k_size_xx = max(int(previous_weights_size[2]), int(new_weights_size[2]))
                    # Dimension 3: Kernel Size YY-Axis
                    max_k_size_yy = max(int(previous_weights_size[3]), int(new_weights_size[3]))

                    # Generate a RandN Torch Tensor that has this Max 4 Dimensions' Sizes
                    randn_weights_tensor = torch.randn(max_nr_out_channels, max_nr_in_channels, max_k_size_xx, max_k_size_yy)

                    # Fill this tensor with the maximum possible values for every dimension with the previous_weights
                    # Get the Previous Weights Dimensions' Sizes
                    previous_weights_out_channels = int(previous_weights_size[0])
                    previous_weights_in_channels = int(previous_weights_size[1])
                    previous_weights_k_size_xx = int(previous_weights_size[2])
                    previous_weights_k_size_yy = int(previous_weights_size[3])

                    # Fill the randn tensor with the maximum possible values
                    randn_weights_tensor[:previous_weights_out_channels,
                                         :previous_weights_in_channels,
                                         :previous_weights_k_size_xx,
                                         :previous_weights_k_size_yy] = previous_weights[:previous_weights_out_channels,
                                                                        :previous_weights_in_channels,
                                                                        :previous_weights_k_size_xx,
                                                                        :previous_weights_k_size_yy]



                    # Transfer the maximum possible values of the randn tensor for the new weights tensor
                    # Get the New Weights Dimensions' Sizes
                    new_weights_out_channels = int(new_weights_size[0])
                    new_weights_in_channels = int(new_weights_size[1])
                    new_weights_k_size_xx = int(new_weights_size[2])
                    new_weights_k_size_yy = int(new_weights_size[3])

                    # Fill the new weights tensor with maximum possible values
                    new_weights[:new_weights_out_channels,
                                :new_weights_in_channels,
                                :new_weights_k_size_xx,
                                :new_weights_k_size_yy] = randn_weights_tensor[:new_weights_out_channels,
                                                                               :new_weights_in_channels,
                                                                               :new_weights_k_size_xx,
                                                                               :new_weights_k_size_yy]


                    # Transfer this to the the pretrained model
                    pretrained_model.convolutional_layers[curr_idx].weight = torch.nn.Parameter(new_weights)


                    # The Bias Tensor only has 1 Dimension, so it is easier to check the minimum size and transfer that ammount
                    # Previous Bias
                    previous_bias = torch.clone(previous_model.convolutional_layers[curr_idx].bias)
                    previous_bias_size = previous_model.convolutional_layers[curr_idx].bias.size()
                    # New Bias
                    new_bias = torch.clone(pretrained_model.convolutional_layers[curr_idx].bias)
                    new_bias_size = pretrained_model.convolutional_layers[curr_idx].bias.size()
                    # Flatten Tensors to avoid dimensional problems
                    previous_bias = previous_bias.view(-1)
                    new_bias = new_bias.view(-1)
                    # Check the minimum size
                    min_size = min(previous_bias.size(0), new_bias.size(0))
                    # Transfer Biases
                    new_bias[0:min_size] = previous_bias[0:min_size]
                    # Reshape New Bias Tensor
                    new_bias = new_bias.view(new_bias_size)
                    # Transfer this to the pretrained model
                    pretrained_model.convolutional_layers[curr_idx].bias = torch.nn.Parameter(new_bias)


                    # Update the conv_idx variable to the next
                    conv_idx +=1

                # Update curr_idx to go through
                curr_idx += 1


        # Then, check fc-layers
        # Create a list with the fc-blocks of each solution
        fc_layers_sols = [previous_best_solution[1], new_candidate_solution[1]]

        # Check the solution which has the lower number of layers
        nr_of_fc_layers = [np.shape(fc_layers_sols[0])[0], np.shape(fc_layers_sols[1])[0]]
        limitant_sol_idx = np.argmin(nr_of_fc_layers)

        # The weights are going to be copied from the number of layers equal to the the number of layers of limitant_sol_idx
        with torch.no_grad():
            curr_idx = 0
            fc_idx = 0
            # We iterate through the number of layers equal to the limitant_sol_idx
            while fc_idx < nr_of_fc_layers[limitant_sol_idx]:
                if isinstance(previous_model.fc_layers[curr_idx], nn.Linear):
                    # Access this index and see the size of weight and bias tensors
                    # Weight Tensors
                    # Previous Weights
                    previous_weights = torch.clone(previous_model.fc_layers[curr_idx].weight)
                    previous_weights_size = previous_model.fc_layers[curr_idx].weight.size()
                    # New Weights
                    new_weights = torch.clone(pretrained_model.fc_layers[curr_idx].weight)
                    new_weights_size = pretrained_model.fc_layers[curr_idx].weight.size()
                    # Flatten Tensors to avoid dimension problems
                    previous_weights = previous_weights.view(-1)
                    new_weights = new_weights.view(-1)
                    # Check the minimum size
                    min_size = min(previous_weights.size(0), new_weights.size(0))
                    # Transfer Weights
                    new_weights[0:min_size] = previous_weights[0:min_size]
                    # Reshape New Weights Tensor
                    new_weights = new_weights.view(new_weights_size)
                    # Transfer this to the the pretrained model
                    pretrained_model.fc_layers[curr_idx].weight = torch.nn.Parameter(new_weights)


                    # Apply the Same to the Bias Tensor
                    # Previous Bias
                    previous_bias = torch.clone(previous_model.fc_layers[curr_idx].bias)
                    previous_bias_size = previous_model.fc_layers[curr_idx].bias.size()
                    # New Bias
                    new_bias = torch.clone(pretrained_model.fc_layers[curr_idx].bias)
                    new_bias_size = pretrained_model.fc_layers[curr_idx].bias.size()
                    # Flatten Tensors to avoid dimensional problems
                    previous_bias = previous_bias.view(-1)
                    new_bias = new_bias.view(-1)
                    # Check the minimum size
                    min_size = min(previous_bias.size(0), new_bias.size(0))
                    # Transfer Biases
                    new_bias[0:min_size] = previous_bias[0:min_size]
                    # Reshape New Bias Tensor
                    new_bias = new_bias.view(new_bias_size)
                    # Transfer this to the pretrained model
                    pretrained_model.fc_layers[curr_idx].bias = torch.nn.Parameter(new_bias)


                    # Update the conv_idx variable to the next
                    fc_idx += 1

                # Update curr_idx to go through
                curr_idx += 1



        # Last, but not least, check the fc-label layer
        with torch.no_grad():
            # Access this index and see the size of weight and bias tensors
                # Weight Tensors
                # Previous Weights
                previous_weights = torch.clone(previous_model.fc_labels.weight)
                previous_weights_size = previous_model.fc_labels.weight.size()
                # New Weights
                new_weights = torch.clone(pretrained_model.fc_labels.weight)
                new_weights_size = pretrained_model.fc_labels.weight.size()
                # Flatten Tensors to avoid dimension problems
                previous_weights = previous_weights.view(-1)
                new_weights = new_weights.view(-1)
                # Check the minimum size
                min_size = min(previous_weights.size(0), new_weights.size(0))
                # Transfer Weights
                new_weights[0:min_size] = previous_weights[0:min_size]
                # Reshape New Weights Tensor
                new_weights = new_weights.view(new_weights_size)
                # Transfer this to the the pretrained model
                pretrained_model.fc_labels.weight = torch.nn.Parameter(new_weights)


                # Apply the Same to the Bias Tensor
                # Previous Bias
                previous_bias = torch.clone(previous_model.fc_labels.bias)
                previous_bias_size = previous_model.fc_labels.bias.size()
                # New Bias
                new_bias = torch.clone(pretrained_model.fc_labels.bias)
                new_bias_size = pretrained_model.fc_labels.bias.size()
                # Flatten Tensors to avoid dimensional problems
                previous_bias = previous_bias.view(-1)
                new_bias = new_bias.view(-1)
                # Check the minimum size
                min_size = min(previous_bias.size(0), new_bias.size(0))
                # Transfer Biases
                new_bias[0:min_size] = previous_bias[0:min_size]
                # Reshape New Bias Tensor
                new_bias = new_bias.view(new_bias_size)
                # Transfer this to the pretrained model
                pretrained_model.fc_labels.bias = torch.nn.Parameter(new_bias)

        return pretrained_model


    # Function: Review Mutation Method
    def apply_mutation(self, alive_solutions_list):
        # Create a mutated solutions list to append solutions
        mutated_solutions_list = list()

        # First, we iterate through the alive solutions list
        for solution in alive_solutions_list:
            # Create a copy of the solution to mutate
            # This way, after the rebuild of the solution we can see if the solution is workable or not
            # If not, we stay with the original solution
            _solution = self.copy_solution(sol=solution)

            # We first check convolutional blocks
            # We create a mask of numbers in interval [0, 1) for the conv-block
            conv_block_mask = torch.rand_like(_solution[0])
            conv_block_mask = conv_block_mask >= self.mutation_rate

            # Iterate through conv-block and evaluate mutation places
            for layer_idx, layer in enumerate(_solution[0]):
                layer_mask = conv_block_mask[layer_idx]

                # Iterate through layer mask and see the places where we have to promote mutation
                for where_mutated, is_mutated in enumerate(layer_mask):

                    # Check mutation places first
                    if where_mutated == 0:
                        if is_mutated == True:
                            # Mutation happens in conv-filters
                            nr_filters = random.choice(utils.conv_nr_filters)
                            _solution[0][layer_idx][where_mutated] = nr_filters

                    elif where_mutated == 1:
                        if is_mutated == True:
                            # Mutation happens in conv_kernel_sizes
                            # max_kernel_size = min(curr_tensor.size()[2:])
                            # allowed_conv_kernel_size = utils.conv_kernel_size[utils.conv_kernel_size <= max_kernel_size]
                            # kernel_size = random.choice(allowed_conv_kernel_size)
                            kernel_size = random.choice(utils.conv_kernel_size)
                            _solution[0][layer_idx][where_mutated] = kernel_size

                    elif where_mutated == 2:
                        if is_mutated == True:
                            # Mutation happens in conv_activ_functions
                            activ_function = random.randint(0, len(utils.conv_activ_functions)-1)
                            _solution[0][layer_idx][where_mutated] = activ_function

                    elif where_mutated == 3:
                        if is_mutated == True:
                            # Mutation happens in conv_drop_rates
                            drop_out = random.uniform(utils.conv_drop_out_range[0], utils.conv_drop_out_range[1])
                            _solution[0][layer_idx][where_mutated] = drop_out

                    else:
                        if is_mutated == True:
                            # Mutation happens in conv_pool_types
                            # max_kernel_size = min(curr_tensor.size()[2:])
                            # if max_kernel_size < 2:
                                # pool = 0
                            # else:
                                # pool = random.randint(0, len(utils.conv_pooling_types)-1)
                            
                            pool = random.randint(0, len(utils.conv_pooling_types)-1)
                            _solution[0][layer_idx][where_mutated] = pool

            
            # We now check the fc-block
            # We create a mask of numbers in interval [0, 1) for the fc-block
            fc_block_mask = torch.rand_like(_solution[1])
            fc_block_mask = fc_block_mask >= self.mutation_rate

            # Iterate through fc-block and evaluate mutation places
            for layer_idx, layer in enumerate(_solution[1]):
                layer_mask = fc_block_mask[layer_idx]

                # Iterate through layer mask and see the places where we have to promote mutation
                for where_mutated, is_mutated in enumerate(layer_mask):
                    if is_mutated == True:
                        if where_mutated == 0:
                            # Mutation happens in fc_neurons
                            nr_neurons = random.randint(utils.fc_nr_neurons_range[0], utils.fc_nr_neurons_range[1])
                            _solution[1][layer_idx][where_mutated] = nr_neurons

                        elif where_mutated == 1:
                            # Mutation happens in fc_activ_functions
                            activ_function = random.randint(0, len(utils.fc_activ_functions) - 1)
                            _solution[1][layer_idx][where_mutated] = activ_function

                        else:
                            # Mutation happens in fc_drop_rates
                            drop_out = random.uniform(utils.fc_drop_out_range[0], utils.fc_drop_out_range[1])
                            _solution[1][layer_idx][where_mutated] = drop_out


            # Last, we check the learning rate
            # We also create a mask for the learning rate
            if torch.rand_like(_solution[2]) >= self.mutation_rate:
                # Mutation happens in the learning rate
                _solution[2] = torch.tensor([random.choice(utils.learning_rate)])


            # Finally, we append the mutated solution to the mutated solutions list
            mutated_solutions_list.append(_solution)


        return mutated_solutions_list

    # Function: Solution Selection Function
    def solution_selection(self, s_population, s_fitnesses):
        # Obtain the total sum of fitnesses
        fitnesses_sum = np.sum(s_fitnesses)

        # Generate probabilities
        f_probabs = [f/fitnesses_sum for f in s_fitnesses]

        # Choose the best solutions
        # This list was created to avoid tensor errors
        s_population_indices = [i for i in range(len(f_probabs))]
        # We select the indices based on probabilities 
        most_fit_solutions = np.random.choice(a=s_population_indices, size=len(s_population_indices), replace=True, p=f_probabs)

        # Create empty list for the most fit solutions and assign in agreement with the indices obtained
        most_fit_solutions = [self.copy_solution(s_population[s]) for s in most_fit_solutions]

        return most_fit_solutions

    # Function: Repair Solution, to repair "damaged" chromossomes after crossover and mutation
    def repair_solution(self, solution):
        solution = self.copy_solution(solution)

        convolutional_layers = solution[0]
        fully_connected_layers = solution[1]
        learning_rate = solution[2]

        # Block Sizes
        nr_conv_layers = convolutional_layers.size()[0]

        # Process Input Shape
        channels = self.input_shape[0]
        rows = self.input_shape[1]
        columns = self.input_shape[2]

        # Create Convolutional Block
        curr_tensor = torch.rand((1, channels, rows, columns))

        for layer in range(nr_conv_layers):

            # Column 1: Conv-Kernel Sizes
            max_kernel_size = min(curr_tensor.size()[2:])
            curr_kernel_size = convolutional_layers[layer][1].item()

            if curr_kernel_size > max_kernel_size:
                curr_kernel_size = max(utils.conv_kernel_size[utils.conv_kernel_size <= max_kernel_size])

            convolutional_layers[layer][1] = curr_kernel_size

            # Update curr_tensor
            curr_tensor = nn.Conv2d(in_channels=int(curr_tensor.size()[1]), out_channels=int(convolutional_layers[layer][0].item()),
                                    kernel_size=int(convolutional_layers[layer][1].item()))(curr_tensor)

            # Column 4: Conv-Pool Layer Types
            max_kernel_size = min(curr_tensor.size()[2:])
            pool = convolutional_layers[layer][4].item()

            if max_kernel_size < 2:
                pool = 0

            convolutional_layers[layer][4] = pool

            # Update curr_tensor
            curr_tensor = utils.conv_pooling_types[int(convolutional_layers[layer][4].item())](curr_tensor)

        return [convolutional_layers, fully_connected_layers, learning_rate]

    # Function: Crossover Method
    def apply_crossover(self, sol1, sol2):
        # Copy solutions first
        sol1 = self.copy_solution(sol1)
        sol2 = self.copy_solution(sol2)

        # Conv
        conv_layers_sol1 = sol1[0]
        conv_layers_sol2 = sol2[0]

        nr_conv_layers_sol1 = conv_layers_sol1.size()[0]
        nr_conv_layers_sol2 = conv_layers_sol2.size()[0]

        cp = random.randint(0, min(nr_conv_layers_sol1, nr_conv_layers_sol2))

        sol1_aux = conv_layers_sol1[cp:]
        sol2_aux = conv_layers_sol2[cp:]

        conv_layers_sol1 = conv_layers_sol1[:cp]
        conv_layers_sol2 = conv_layers_sol2[:cp]

        conv_layers_sol1 = torch.cat((conv_layers_sol1, sol2_aux), dim=0)
        conv_layers_sol2 = torch.cat((conv_layers_sol2, sol1_aux), dim=0)

        sol1[0] = conv_layers_sol1
        sol2[0] = conv_layers_sol2

        # fc

        fc_layers_sol1 = sol1[1]
        fc_layers_sol2 = sol2[1]

        nr_fc_layers_sol1 = fc_layers_sol1.size()[0]
        nr_fc_layers_sol2 = fc_layers_sol2.size()[0]

        cp = random.randint(0, min(nr_fc_layers_sol1, nr_fc_layers_sol2))

        sol1_aux = fc_layers_sol1[cp:]
        sol2_aux = fc_layers_sol2[cp:]

        fc_layers_sol1 = fc_layers_sol1[:cp]
        fc_layers_sol2 = fc_layers_sol2[:cp]

        fc_layers_sol1 = torch.cat((fc_layers_sol1, sol2_aux), dim=0)
        fc_layers_sol2 = torch.cat((fc_layers_sol2, sol1_aux), dim=0)

        sol1[1] = fc_layers_sol1
        sol2[1] = fc_layers_sol2

        return sol1, sol2

    # Function: Fitness
    def solution_fitness(self, solution_acc):
        # The fitness is the solution accuracy
        s_fitness = solution_acc

        return s_fitness