import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.animation as animation


class Perceptron(object):

    def __init__(self, learning_rate):

        self.weights = [12.0, 4.2]
        self.lr = learning_rate  # ordinary 0 < lr < 1
        self.bias = 1.0

    def __iter__(self):
        return self

    def __next__(self):

        return self.weights[0], self.weights[1]

    def prediction(self, input_x):

        z = (self.weights[0] * input_x[0] + self.weights[1] * input_x[1] + self.bias)
        activation = get_activation(z)
        
        return activation

    def train(self, input_x, teaching_label):
        
        prediction = self.prediction(input_x)

        for i in range(len(input_x)):
            update_weight = self.weights[i] + self.lr * (teaching_label - prediction) * input_x[i]
            self.weights[i] = update_weight
            
        update_bias = self.bias + self.lr*(teaching_label - prediction)
        self.bias = update_bias


def get_activation(z):

    return int(z >= 0) # return True=1 False=0


def read_file(path):

    file = open(path, "r")

    input_x = []
    teaching_label = []

    for line in file:

        x = line.split()
        input_x.append([float(x[0]), float(x[1])])
        teaching_label.append(float(x[2]))

    file.close()

    return input_x, teaching_label


def check_validity_of_weights(input_x, teaching_label, object1):

    total = 0
    for i in range(len(teaching_label)):

        if object1.prediction(input_x[i]) == teaching_label[i]:
            total += 1

    accuracy_rate = total/len(teaching_label)*100

    print("\n")
    print("validity of weights = {}%".format(accuracy_rate))
    print("\n")


def display_diagram(path, wvec):

    plot_data = np.loadtxt(path)
    x_fig = np.arange(-5, 10, 0.1)
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ims = []
    sns.set()
    #plt.grid(color='gray')
    plt.axis(xmin=-3, xmax=13, ymin=-3, ymax=10)

    # plot
    for w in wvec:
        y_fig = [-(w[0]/w[1])*xi-(w[2]/w[1]) for xi in x_fig]
        plt.scatter(plot_data[:, 0], plot_data[:, 1], c=plot_data[:, 2], s=20, cmap='RdYlBu')
        ims.append(plt.plot(x_fig, y_fig))
    for i in range(10):
        ims.append(plt.plot(x_fig, y_fig))
    ani = animation.ArtistAnimation(fig, ims, interval=300)
    plt.show(ani)


def main():

    wvec = []
    input_x_training_data, teaching_labels_training_data = read_file("./data/training_data.txt")
    input_x_test_data, teaching_labels_test_data = read_file("./data/test_data.txt")

    object1 = Perceptron(0.01)
    check_validity_of_weights(input_x_test_data, teaching_labels_test_data, object1)

    # train
    epoch = 30
    for i in range(epoch):
        weights = []
        for input_x, teaching_label in zip(input_x_training_data, teaching_labels_training_data):
            object1.train(input_x, teaching_label)

        print(object1.weights, object1.bias)
        weight1, weight2 = next(object1)
        weights.extend([weight1, weight2, object1.bias])
        wvec.append(np.array(weights))

    display_diagram("./data/training_data.txt", wvec)
    check_validity_of_weights(input_x_test_data, teaching_labels_test_data, object1)


if __name__ == '__main__':

    main()

        

