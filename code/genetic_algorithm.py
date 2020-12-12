import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from code.solution import Solution
from code.model import Model
import time
import copy
from code.utilities import utils
import random

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
    def __init__(self, input_shape, number_of_labels, size_of_population, nr_of_generations, mutation_rate, percentage_of_best_fit,
                 survival_rate_of_less_fit, start_phase, end_phase, initial_chromossome_length):


        # Dataset Variables
        self.input_shape = input_shape
        self.number_of_labels = number_of_labels

        # Initialize variables
        self.size_of_population = size_of_population
        self.nr_of_generations = nr_of_generations
        self.mutation_rate = mutation_rate
        self.percentage_of_best_fit = percentage_of_best_fit
        self.survival_rate_of_less_fit = survival_rate_of_less_fit

        # Phase Variables
        self.start_phase = start_phase
        self.current_phase = start_phase
        self.end_phase = end_phase

        # Chromossome Length Variables
        self.initial_chromossome_length = initial_chromossome_length
        self.current_chromossome_length = initial_chromossome_length

        # Some important dictionaries to be used
        # Activation Functions Dict
        self.inv_activ_functions = dict()
        self.inv_activ_functions[0] = 'none'
        self.inv_activ_functions[1] = 'relu'
        self.inv_activ_functions[2] = 'tanh'

        # Pooling Types Dict
        self.inv_pooling_types = dict()
        self.inv_pooling_types[0] = 'none'
        self.inv_pooling_types[1] = 'max'
        self.inv_pooling_types[2] = 'avg'

    # Generate Solutions
    def generate_candidate_solutions(self):
        # Create list to append solutions
        list_of_candidate_solutions = list()

        # Go through the size of the population
        # Initialize current p
        p = 0
        # We have to build p == size_of_population solutions
        while p < self.size_of_population:
            # Initialize empty lists of solution parameters
            conv_filters = list()
            conv_kernel_sizes = list()
            conv_activ_functions = list()
            conv_drop_rates = list()
            conv_pool_types = list()
            fc_neurons = list()
            fc_activ_functions = list()
            fc_drop_rates = list()

            # We have to build a solution with the current chromossome length
            sol_c_length = 0
            while sol_c_length < self.current_chromossome_length:
                # Decide if  we have convolutional layers
                add_conv_layer = np.random.choice(a=[True, False])
                if add_conv_layer:
                    # Conv Filter
                    c_filter = np.random.choice(a=[8, 16, 32, 64, 128, 256, 512])
                    conv_filters.append(c_filter)
                    # Conv Kernel Size
                    c_kernel = np.random.choice(a=[1, 3, 5, 7, 9])
                    conv_kernel_sizes.append(c_kernel)
                    # Conv Activation Functions
                    c_activ_fn = self.inv_activ_functions[np.random.choice(a=[0, 1, 2])]
                    conv_activ_functions.append(c_activ_fn)
                    # Conv Dropout Rate
                    c_drop_rate = np.random.uniform(low=0.0, high=1.0)
                    conv_drop_rates.append(c_drop_rate)
                    # Conv Pool Types
                    # c_pool_tp = inv_pooling_types[0]
                    c_pool_tp = self.inv_pooling_types[np.random.choice(a=[0, 1, 2])] # TODO Check if this works with our validation routine
                    conv_pool_types.append(c_pool_tp)
                    # Update current c_length
                    sol_c_length += 1


                # Otherwise, we add a FC-Layer
                else:
                    # FC Neurons
                    fc_out_neuron = np.random.uniform(low=1.0, high=100)
                    fc_neurons.append(fc_out_neuron)
                    # FC Activation Function
                    fc_activ_fn = self.inv_activ_functions[np.random.choice(a=[0, 1, 2])]
                    fc_activ_functions.append(fc_activ_fn)
                    # FC Dropout Rate
                    fc_drop = np.random.uniform(low=0.0, high=1.0)
                    fc_drop_rates.append(fc_drop)
                    # Update current c_length
                    sol_c_length += 1

            # Decide the learning-rate
            learning_rate = np.random.choice(a=[0.001, 0.0001, 0.00001])

            # Build solution
            solution = Solution(
                conv_filters=conv_filters,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_activ_functions=conv_activ_functions,
                conv_drop_rates=conv_drop_rates,
                conv_pool_types=conv_pool_types,
                fc_neurons=fc_neurons,
                fc_activ_functions=fc_activ_functions,
                fc_drop_rates=fc_drop_rates,
                learning_rate=learning_rate
            )

            # Test solution with Model to see if it is a viable solution
            try:
                _ = Model(self.input_shape, self.number_of_labels, solution.get_solution_matrix())
            
            # If it goes wrong, p value stays the same
            except:
                p = p            
            
            # If it goes OK, we can append the solution to our solution candidates
            else:
                # Append this solution to the list of candidate solutions
                list_of_candidate_solutions.append(solution)
                # Update current p
                p += 1

        return list_of_candidate_solutions

    # TODO: Normalize Data (compute data mean and std manually):
    def normalize_data(self):
        mnist_mean = [0.1307]
        mnist_std = [0.3081]

        fashion_mnist_mean = [0.2860]
        fashion_mnist_std = [0.3530]

        cifar10_mean = [0.4914, 0.4822, 0.4465]
        cifar10_std = [0.2470, 0.2435, 0.2616]
        return None

    # TODO: Training Method
    def train(self):
        # 1) create model with a given solution
        # 2) train model
        # 3) calculate model cost

        # data // TODO:  We can erase this since these variables are now class variables 
        # input_shape = [28, 28, 1]
        # number_of_labels = 10

        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307],
                                                                                          std=[0.3081])])

        mnist_data = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=train_transform)

        data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=32, shuffle=True, num_workers=4)

        # create models
        models = []

        # Create candidate solutions for the present phase and generations 
        # (this list will be usefull to the next steps such as mutation and crossover)
        gen_candidate_solutions = self.generate_candidate_solutions()

        for candidate in gen_candidate_solutions:
            # TODO: Solution structure changed
            # candidate = candidate.build_solution()
            models.append(Model(self.input_shape, self.number_of_labels, candidate.get_solution_matrix()))

        # loss
        loss = nn.CrossEntropyLoss()

        # select gpu if possible
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_start = time.time()

        # train
        for model in models:
            print('training model')

            model = model.to(device)
            model.train()
            optim = torch.optim.Adam(model.parameters(), lr=model.learning_rate) # TODO Check if this works

            every_x_minibatches = 100

            for epoch in range(5):

                running_loss = 0.0

                start = time.time()

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

                    # print statistics
                    running_loss += loss_value.item()
                    if (i + 1) % every_x_minibatches == 0:
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / every_x_minibatches))
                        running_loss = 0.0

                end = time.time()

                print('epoch time: ' + str(end - start))

                # torch.save(model.state_dict(), 'model.pth')

            print('Finished Training')
            model = model.cpu()

        print('Total time: ' + str(time.time()-model_start))
        # total time on my pc with gpu 1129 seg ~ 18 min. (specs: ryzen 7 3700x, rtx 2070S, 32gb ram 3600mhz)


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
            _solution = copy.deepcopy(solution)

            # Generate a random number between 0-1 to compare against the mutation rate
            mutation_proba = np.random.uniform(low=0.0, high=1.0)
            
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
        # The solution cost is the solution loss minus the solution accuracy: this way we penalise the loss value and reward the accuracy
        # Since we want to convert this into a maximisation problem, we multiply the value of the solution cost by -1
        solution_cost = -1 * (solution_loss - solution_acc)

        return solution_cost