import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from code.solution import Solution
from code.model import Model
import time


# Define the Genetic Algorithm Class
class GeneticAlgorithm:
    def __init__(self, size_of_population, nr_of_generations, mutation_rate, percentage_of_best_fit,
                 survival_rate_of_less_fit, start_phase, end_phase, initial_chromossome_length, random_seed=42):
        # Initialize random seeds
        # NumPy
        np.random.seed(random_seed)

        # PyTorch
        torch.manual_seed(random_seed)

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

    # Generate Solutions
    def generate_candidate_solutions(self):
        # Create list to append solutions
        list_of_candidate_solutions = list()

        inv_activ_functions = dict()
        inv_activ_functions[0] = 'none'
        inv_activ_functions[1] = 'relu'
        inv_activ_functions[2] = 'tanh'

        inv_pooling_types = dict()
        inv_pooling_types[0] = 'none'
        inv_pooling_types[1] = 'max'
        inv_pooling_types[2] = 'avg'

        # Go through the size of the population
        for _ in range(self.size_of_population):
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
                    c_activ_fn = inv_activ_functions[np.random.choice(a=[0, 1, 2])]
                    conv_activ_functions.append(c_activ_fn)
                    # Conv Dropout Rate
                    c_drop_rate = np.random.uniform(low=0.0, high=1.0)
                    conv_drop_rates.append(c_drop_rate)
                    # Conv Pool Types
                    c_pool_tp = inv_pooling_types[
                        0]  # inv_pooling_types[np.random.choice(a=[0, 1, 2])]  # TODO change this back
                    conv_pool_types.append(c_pool_tp)
                    # Update current c_length
                    sol_c_length += 1


                # Otherwise, we add a FC-Layer
                else:
                    # FC Neurons
                    fc_out_neuron = np.random.uniform(low=1.0, high=100)
                    fc_neurons.append(fc_out_neuron)
                    # FC Activation Function
                    fc_activ_fn = inv_activ_functions[np.random.choice(a=[0, 1, 2])]
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

            # TODO: Test solution with Model to see if it is a viable solution
            

            # Append this solution to the list of candidate solutions
            list_of_candidate_solutions.append(solution)

        return list_of_candidate_solutions

    # TODO: Normalize Data (compute data mean and std manually):
    def normalize_data(self):
        mnist_mean = [0.1307]
        mnist_std = [0.3081]

        fashion_mnist_mean = [0.2860]
        fashion_mnist_std = [0.3530]

        cifar10_mean = [0.4914, 0.4822, 0.4465]
        cifar10_std = [0.2470, 0.2435, 0.2616]
        return

    # TODO: Training Method
    def train(self):
        # 1) create model with a given solution
        # 2) train model
        # 3) calculate model cost

        # data
        input_shape = [28, 28, 1]
        number_of_labels = 10

        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307],
                                                                                          std=[0.3081])])

        mnist_data = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=train_transform)

        data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=32, shuffle=True, num_workers=4)

        # create models
        models = []

        for candidate in self.generate_candidate_solutions():
            models.append(Model(input_shape, number_of_labels, candidate.get_solution_matrix()))

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
            optim = torch.optim.Adam(model.parameters()) # TODO model learning rate

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
    def apply_mutation(self):
        # TODO: Randomly change parameters inside the solution
        pass

    # TODO: Crossover Method
    def apply_crossover(self):
        # TODO: Crossover between solution (random layers to hybrid); pay attention to the number of conv layers and fc layers of mum and dad
        pass

    # Fitness Function
    def solution_fitness(self, solution_acc, solution_loss):
        # The solution cost is the solution loss minus the solution accuracy: this way we penalise the loss value and reward the accuracy
        # Since we want to convert this into a maximisation problem, we multiply the value of the solution cost by -1
        solution_cost = -1 * (solution_loss - solution_acc)

        return solution_cost