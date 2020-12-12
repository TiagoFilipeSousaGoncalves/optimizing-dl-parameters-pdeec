# Imports
import numpy as np
import time
import copy
import random

# Torch Imports
import torch
import torch.nn as nn

# Project Imports
from code.model import Model
from code.utilities import utils
from code.datasets import get_mnist_loader

# Sklearn Imports
import sklearn.metrics as sklearn_metrics

# Set random seed value so we a have a reproductible work
random_seed = 42

# Initialize random seeds
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


# input_shape is [C, H, W]
def generate_random_solution(chromossome_length, input_shape):
    # Block Sizes
    nr_conv_layers = np.random.randint(0, chromossome_length + 1)
    nr_fc_layers = chromossome_length - nr_conv_layers

    # Process Input Shape
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


def copy_solution(sol):
    return [sol[0].clone(), sol[1].clone(), sol[2].clone()]


# Define the Genetic Algorithm Class
class GeneticAlgorithm:
    def __init__(self, input_shape, number_of_labels, size_of_population, nr_of_generations, mutation_rate,
                 percentage_of_best_fit, survival_rate_of_less_fit, start_phase, end_phase, initial_chromossome_length, nr_of_epochs=5):

        # Dataset Variables
        self.input_shape = input_shape
        self.number_of_labels = number_of_labels

        # Initialize variables
        self.size_of_population = size_of_population
        self.nr_of_generations = nr_of_generations
        self.mutation_rate = mutation_rate
        self.percentage_of_best_fit = percentage_of_best_fit
        self.survival_rate_of_less_fit = survival_rate_of_less_fit
        self.nr_of_epochs = nr_of_epochs

        # Phase Variables
        self.start_phase = start_phase
        self.current_phase = start_phase
        self.end_phase = end_phase

        # Chromossome Length Variables
        self.initial_chromossome_length = initial_chromossome_length
        self.current_chromossome_length = initial_chromossome_length

    # Generate Solutions
    def generate_candidate_solutions(self):
        # Create list to append solutions
        list_of_candidate_solutions = []

        # Go through the size of the population
        # Initialize current p
        p = 0
        # We have to build p == size_of_population solutions
        while p < self.size_of_population:

            # Append this solution to the list of candidate solutions
            list_of_candidate_solutions.append(generate_random_solution(self.current_chromossome_length, self.input_shape))
            # Update current p
            p += 1

        return list_of_candidate_solutions

    # TODO: Review Training Method
    def train(self):
        # Data will always be the same, so we can read it in the beginning of the loop
        data_loader = get_mnist_loader(32)

        # Evaluate the current phase agains the maximum number of phases
        while self.current_phase < self.end_phase:
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
                
                # If not, we are generating new solutions; we obtain new ones by crossover and mutation
                else:
                    # TODO: Apply crossover between best solutions of the previous generation until you achieve
                    # the size of the populations

                    # TODO: Apply random mutations to the population

                    pass

                # Create models list
                models = []
                
                # Create models from these candidate solution
                for candidate in gen_candidate_solutions:
                    models.append(Model(self.input_shape, self.number_of_labels, candidate))
                
                # Choose loss function; here we use CrossEntropy
                loss = nn.CrossEntropyLoss()

                # Select device: GPU or CPU
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

                    # TODO: Delete this
                    # FLAG for print statistics
                    # every_x_minibatches = 100

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

                            # TODO: Erase this
                            # if (i + 1) % every_x_minibatches == 0:
                                # print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / every_x_minibatches}')
                                # running_loss = 0.0

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
                # total time on my pc with gpu 1129 seg ~ 18 min. (specs: ryzen 7 3700x, rtx 2070S, 32gb ram 3600mhz)

                # Evaluate Generations Solutions Fitness
                generation_solutions_fitness = [self.solution_fitness(r[0], r[1]) for r in generation_models_results]
                print(generation_solutions_fitness)



    # TODO: Thread Training Solution (this would only give a performance boost using different processes, not threads, i think. I dont know how hard it is to implement,
    #  because sharing memory between processes can be a pain sometimes. Even if we implement it this would only give a performance boost if the gpu can train multiple
    #  models simultaneously without reaching its parallel processing capability. I think this should be the last thing to implement)
    def thread_training(self):
        pass

    # TODO: Transfer learning
    def transfer_learning(self):
        pass

    # TODO: Mutation Method
    def apply_mutation(self, alive_solutions_list):
        # Create a mutated solutions list to append solutions
        mutated_solutions_list = list()
        
        # First, we iterate through the alive solutions list
        for solution in alive_solutions_list:
            # Create a copy of the solution to mutate
            # This way, after the rebuild of the solution we can see if the solution is workable or not
            # If not, we stay with the original solution
            _solution = copy_solution(sol=solution)

            # TODO: Erase after review
            # Generate a random number between 0-1 to compare against the mutation rate
            # mutation_proba = np.random.uniform(low=0.0, high=1.0)

            # We first check convolutional blocks
            # We create a mask of numbers in interval [0, 1) for the conv-block
            conv_block_mask = torch.rand_like(_solution[0])
            conv_block_mask = conv_block_mask >= self.mutation_rate

            # Create a Tensor to evaluate feasibility while creating the mutated solution
            curr_tensor = torch.rand((1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            
            # Iterate through conv-block and evaluate mutation places
            for layer_idx, layer in enumerate(_solution[0]):
                layer_mask = conv_block_mask[layer_idx]

                # Iterate through layer mask and see the places where we have to promote mutation
                for where_mutated, is_mutated in enumerate(layer_mask):
                    # We have to update our curr_tensor before going into Column 2 mutations
                    if where_mutated == 2:
                        curr_tensor = nn.Conv2d(in_channels=curr_tensor.size()[1], out_channels=_solution[0][layer_idx][0], kernel_size=_solution[0][layer_idx][1])(curr_tensor)

                    if is_mutated == True:
                        if where_mutated == 0:
                            # Mutation happens in conv-filters
                            _solution[0][layer_idx][where_mutated] = random.choice(utils.conv_nr_filters)
                        
                        elif where_mutated == 1:
                            # Mutation happens in conv_kernel_sizes
                            max_kernel_size = min(curr_tensor.size()[2:])
                            allowed_conv_kernel_size = utils.conv_kernel_size[utils.conv_kernel_size <= max_kernel_size]
                            kernel_size = random.choice(allowed_conv_kernel_size)
                            _solution[0][layer_idx][where_mutated] = kernel_size

                        elif where_mutated == 2:
                            # Mutation happens in conv_activ_functions
                            activ_function = random.randint(0, len(utils.conv_activ_functions)-1)
                            _solution[0][layer_idx][where_mutated] = activ_function
                        
                        elif where_mutated == 3:
                            # Mutation happens in conv_drop_rates
                            drop_out = random.uniform(utils.conv_drop_out_range[0], utils.conv_drop_out_range[1])
                            _solution[0][layer_idx][where_mutated] = drop_out

                        else:
                            # Mutation happens in conv_pool_types
                            max_kernel_size = min(curr_tensor.size()[2:])
                            if max_kernel_size < 2:
                                pool = 0
                            else:
                                pool = random.randint(0, len(utils.conv_pooling_types)-1)
                            
                            _solution[0][layer_idx][where_mutated] = pool
                    
                    # Update curr_tensor after Column 4 mutations
                    curr_tensor = utils.conv_pooling_types[pool](curr_tensor)

            
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
                            _solution[1][layer_idx][where_mutated]
                            pass

                        elif where_mutated == 1:
                            # Mutation happens in fc_activ_functions
                            _solution[1][layer_idx][where_mutated]
                            pass

                        else:
                            # Mutation happens in fc_drop_rates
                            _solution[1][layer_idx][where_mutated]
                            pass
            
            
            # Last, we check the learning rate
            # We also create a mask for the learning rate
            if torch.rand_like(_solution[2]) >= self.mutation_rate:
                # Mutation happens in the learning rate
                _solution[2] = torch.tensor([random.choice(utils.learning_rate)])
            

            # If it's bigger than the defined mutation rate we apply a mutation
            if mutation_proba >= self.mutation_rate:
                # TODO: Review where should we apply mutation
                # For now, let's assume that we can randomly choice where to apply these
                # TODO: Create a matrix with random numbers (0-1) for each parameter with the probas for each param 
                # of each layer
                where_to_mutate = np.random.choice(a=[0, 1, 2]) # TODO: This goes out!
                # 0 - TODO: Apply on the Conv-Layers
                if where_to_mutate == 0:
                    # Check the size of the convolutional layers block
                    conv_block_len = _solution.convolutional_layers.size(0)

                    # Choose a random layer to apply a mutation
                    # We create a list with the indices first
                    conv_block_layers_indices = [i for i in range(conv_block_len)]
                    # We choose a layer to apply mution based on the index
                    c_layer_idx = np.random.choice(a=conv_block_layers_indices)


                    # Now, we need to choose which of the parameters are we going to mutate
                    conv_layer_params_to_change = ["conv_filters", "conv_kernel_sizes", "conv_activ_functions", "conv_drop_rates", "conv_pool_types"]
                    c_param_change = np.random.choice(a=conv_layer_params_to_change)

                    # Check the parameter to change
                    # "conv_filters"
                    if c_param_change == "conv_filters":
                        # We change the number of c_filter of the c_layer_idx 
                        _solution.conv_filters[c_layer_idx] = np.random.choice(a=[8, 16, 32, 64, 128, 256, 512])  
                    
                    # "conv_kernel_sizes"
                    elif c_param_change == "conv_kernel_sizes":
                        # We change the kernel size of the c_layer_idx
                        _solution.conv_kernel_sizes[c_layer_idx] = np.random.choice(a=[1, 3, 5, 7, 9])
                    
                    # "conv_activ_functions"
                    elif c_param_change == "conv_activ_functions":
                        # We change the activation function of the c_layer_idx
                        _solution.conv_activ_functions[c_layer_idx] = self.inv_activ_functions[np.random.choice(a=[0, 1, 2])]
                    
                    # "conv_drop_rates"
                    elif c_param_change == "conv_drop_rates":
                        # We change the dropout rate of the c_layer_idx
                        _solution.conv_drop_rates[c_layer_idx] = np.random.uniform(low=0.0, high=1.0)
                    
                    # Otherwise, c_param_change == "conv_pool_types"
                    else:
                        # We change the pooling type of the c_layer_idx
                        _solution.conv_pool_types[c_layer_idx] = self.inv_pooling_types[np.random.choice(a=[0, 1, 2])]

                
                # 1 - TODO: Apply on the FC-Layers
                elif where_to_mutate == 1:
                    # Check the size of the FC block
                    fc_block_len = _solution.fully_connected_layers.size(0)
                    
                    # Choose a random layer to apply mutation
                    # We create a list with the indices first
                    fc_block_indices = [i for i in range(fc_block_len)]
                    # We choose a layer to apply mutation based on the index
                    fc_layer_idx = np.random.choice(a=fc_block_indices)

                    # Now, we need to choose which of the parameters are we going to mutate
                    fc_layer_params_to_change = ["fc_neurons", "fc_activ_functions", "fc_drop_rates"]
                    fc_param_change = np.random.choice(a=fc_layer_params_to_change)


                    # Check the parameter to change
                    # "fc_neurons"
                    if fc_param_change == "fc_neurons":
                        # We change the number of out-neurons of the fc_layer_idx
                        _solution.fc_neurons[fc_layer_idx] = np.random.uniform(low=1.0, high=100)
                    
                    # "fc_activ_functions"
                    elif fc_param_change == "fc_activ_functions":
                        # We change the type of activation function of the neuron of the fc_layer_idx
                        _solution.fc_activ_functions[fc_layer_idx] = self.inv_activ_functions[np.random.choice(a=[0, 1, 2])]
                    
                    # "fc_drop_rates"
                    else:
                        # We change the dropout rate of the neuron of the fc_layer_idx
                        _solution.fc_drop_rates[fc_layer_idx] = np.random.uniform(low=0.0, high=1.0)

                
                # 2 - Apply on the learning rate
                else:
                    mutated_lr = np.random.choice(a=[0.001, 0.0001, 0.00001])
                    _solution.learning_rate = mutated_lr

            
            # Test the mutated solutio (_solution) with Model to see if it is a viable solution
            try:
                # TODO: buil_solution function no longer needed
                # _solution = _solution.build_solution()
                _ = Model(self.input_shape, self.number_of_labels, _solution.get_solution_matrix())
            
            # If it goes wrong, we keep the initial solution
            except:
                mutated_solutions_list.append(solution)            
            
            # If it goes OK, we can append the _solution to our mutated solution list
            else:
                # Append this _solution to the list of mutated solutions
                mutated_solutions_list.append(_solution)
            


        # TODO: Review code
        
        return mutated_solutions_list

    # TODO: Crossover Method
    # TODO: Cross-probability
    # TODO: Decide the the survival criteria
    def apply_crossover(self, mutated_solutions_list):
        # TODO: Crossover between solution (random layers to hybrid); pay attention to the number of conv layers and fc layers of mum and dad

        return # new_generation

    # Fitness Function
    def solution_fitness(self, solution_acc, solution_loss):
        # The solution cost is the solution accuracy minus the solution loss: this way we penalise the loss value and reward the accuracy
        # We aim to maximise this value
        s_fitness = solution_acc - solution_loss

        return s_fitness


if __name__ == '__main__':
    ga = GeneticAlgorithm(input_shape=[1, 28, 28], number_of_labels=10, size_of_population=10, nr_of_generations=10, mutation_rate=0.5,
                        percentage_of_best_fit=0.5, survival_rate_of_less_fit=0.5, start_phase=0, end_phase=1, initial_chromossome_length=2)

    ga.train()

