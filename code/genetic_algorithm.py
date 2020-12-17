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
from sklearn.preprocessing import StandardScaler

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
            print(f"Current Training Phase: {self.current_phase}")
            
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
                    gen_candidate_solutions = list()
                    print(f"Generation {current_generation} solutions' crossover applied.")

                    # TODO: Apply random mutations to the population
                    gen_candidate_solutions = self.apply_mutation(alive_solutions_list=most_fit_solutions)
                    print(f"Generation {current_generation} solutions' mutations applied.")

                    # TODO: Repair solutions so we have a complete list of feasible solutions
                    gen_candidate_solutions = [self.repair_solution(s) for s in gen_candidate_solutions]
                    print(f"Generation {current_generation} solutions generated and repaired.")

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
                # TODO: Normalize values before evaluating fitness
                # Create a sklearn scaler
                scaler = StandardScaler()
                
                # Fit scaler to our model results
                scaler.fit(generation_models_results)

                # Convert results
                generation_models_results_scaled = scaler.transform(generation_models_results)


                generation_solutions_fitness = [self.solution_fitness(r[0], r[1], r[2]) for r in generation_models_results_scaled]
                print(generation_solutions_fitness)

                # TODO: Change this after applying the selection rule
                most_fit_solutions = gen_candidate_solutions
                
                # TODO: Updated Generation
                current_generation += 1

            # TODO: Update phase
            self.current_phase += 1


    # TODO: Thread Training Solution (this would only give a performance boost using different processes, not threads, i think. I dont know how hard it is to implement,
    #  because sharing memory between processes can be a pain sometimes. Even if we implement it this would only give a performance boost if the gpu can train multiple
    #  models simultaneously without reaching its parallel processing capability. I think this should be the last thing to implement)
    def thread_training(self):
        pass

    # TODO: Transfer learning
    # TODO: debug: transfer weights from a pretrained network to a new one and check the accuracy
    def transfer_learning(self, previous_model_state_dict, previous_best_solution, new_candidate_solution):
        # Create a Model Instance
        previous_model = Model(self.input_shape, self.number_of_labels, previous_best_solution)
        
        # Load weights
        previous_model.load_state_dict(torch.load(previous_model_state_dict))
        previous_model.eval()

        # Create a Model Instance 
        pretrained_model = Model(self.input_shape, self.number_of_labels, new_candidate_solution)

        # Check conv-layers first
        # Create a list with the conv-blocks of each solution
        conv_layers_sols = [previous_best_solution[0], new_candidate_solution[0]]
        # Check the solution which has the lower number of layers
        limitant_sol_idx = np.argmin([np.shape(conv_layers_sols[0]), np.shape(conv_layers_sols[1])])



        return pretrained_model 

    # TODO: Review Mutation Method
    def apply_mutation(self, alive_solutions_list):
        # Create a mutated solutions list to append solutions
        mutated_solutions_list = list()
        
        # First, we iterate through the alive solutions list
        for solution in alive_solutions_list:
            # Create a copy of the solution to mutate
            # This way, after the rebuild of the solution we can see if the solution is workable or not
            # If not, we stay with the original solution
            _solution = copy_solution(sol=solution)

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
                    
                    # Check mutation places first
                    if where_mutated == 0:
                        if is_mutated == True:
                            # Mutation happens in conv-filters
                            nr_filters = random.choice(utils.conv_nr_filters)
                            _solution[0][layer_idx][where_mutated] = nr_filters
                    
                    elif where_mutated == 1:
                        if is_mutated == True:
                            # Mutation happens in conv_kernel_sizes
                            max_kernel_size = min(curr_tensor.size()[2:])
                            allowed_conv_kernel_size = utils.conv_kernel_size[utils.conv_kernel_size <= max_kernel_size]
                            kernel_size = random.choice(allowed_conv_kernel_size)
                            _solution[0][layer_idx][where_mutated] = kernel_size
                        
                        # Update curr_tensor
                        _nr_filters = int(_solution[0][layer_idx][0].item())
                        _kernel_size = int(_solution[0][layer_idx][where_mutated].item())
                        curr_tensor = nn.Conv2d(in_channels=curr_tensor.size()[1], out_channels=_nr_filters, kernel_size=_kernel_size)(curr_tensor)

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
                            max_kernel_size = min(curr_tensor.size()[2:])
                            if max_kernel_size < 2:
                                pool = 0
                            else:
                                pool = random.randint(0, len(utils.conv_pooling_types)-1)
                            
                            _solution[0][layer_idx][where_mutated] = pool
                        
                        # Update curr_tensor after Column 4 mutations in conv_pool_type
                        _pool = int(_solution[0][layer_idx][where_mutated].item())
                        curr_tensor = utils.conv_pooling_types[_pool](curr_tensor)
            
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

    # TODO selection circle probability
    def solution_selection(self, s_population, s_fitnesses):
        # Create empty list for the most fit solutions
        most_fit_solutions = list()

        return most_fit_solutions

    # Repair Solution Function: to repair "damaged" chromossomes after crossover and mutation
    def repair_solution(self, solution):
        solution = copy_solution(solution)

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
            curr_tensor = nn.Conv2d(in_channels=curr_tensor.size()[1], out_channels=convolutional_layers[layer][0].item(),
                                    kernel_size=convolutional_layers[layer][1].item())(curr_tensor)

            # Column 4: Conv-Pool Layer Types
            max_kernel_size = min(curr_tensor.size()[2:])
            pool = convolutional_layers[layer][4].item()

            if max_kernel_size < 2:
                pool = 0

            convolutional_layers[layer][4] = pool

            # Update curr_tensor
            curr_tensor = utils.conv_pooling_types[convolutional_layers[layer][4].item()](curr_tensor)

        return [convolutional_layers, fully_connected_layers, learning_rate]

    # TODO: Crossover Method
    # TODO: Cross-probability
    # TODO: Decide the the survival criteria
    # TODO: learning rate crossover?
    def apply_crossover(self, sol1, sol2):
        # TODO: Crossover between solution (random layers to hybrid); pay attention to the number of conv layers and fc layers of mum and dad
        # conv
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

        return

    # Fitness Function
    def solution_fitness(self, solution_acc, solution_loss, solution_time):
        # The solution cost is the solution accuracy minus the solution loss: this way we penalise the loss value and reward the accuracy
        # We aim to maximise this value
        # TODO: Review functioning. StandardScaler from sklearn is being applied during training loop
        # TODO: See if it is worthy to take time into account
        s_fitness = (1/solution_time) * (solution_acc - solution_loss)

        return s_fitness


if __name__ == '__main__':
    ga = GeneticAlgorithm(input_shape=[1, 28, 28], number_of_labels=10, size_of_population=2, nr_of_generations=3, mutation_rate=0.5,
                        percentage_of_best_fit=0.5, survival_rate_of_less_fit=0.5, start_phase=0, end_phase=1, initial_chromossome_length=2, nr_of_epochs=1)

    # ga.train()
    # print(ga.repair_solution([torch.tensor([[10, 9, 0, 0, 1], [10, 9, 0, 0, 1], [10, 9, 0, 0, 1], [10, 9, 0, 0, 1], [10, 3, 0, 0, 1]]), torch.tensor([]), torch.tensor([])]))
    a = [torch.tensor([]), torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
          torch.tensor([])]

    b = [torch.tensor([]), torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
          torch.tensor([])]

    ga.apply_crossover(a, b)

    print(a)
    print(b)
