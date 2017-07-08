
class Perceptron(object):

    def __init__(self, learning_rate):

        self.weights = [1, 1]
        self.lr = learning_rate  # ordinary 0<lr<1
        self.bias = 1.0

    def prediction(self, x):

        z =(self.weights[0] * x[0] + self.weights[1]*x[1] + self.bias)
        activation = get_activation(z)
        
        return activation

    def train(self, x, y):
        
        prediction = self.prediction(x)

        for i in range(len(x)):
            update_weight = self.weights[i] + self.lr * (y - prediction) * x[i]
            self.weights[i] = update_weight
            
        update_bias = self.bias + self.lr*(y-prediction)
        self.bias = update_bias
        

def get_activation(z):

    return int(z > 0)


def read_file(path):

    file = open(path, "r")

    x_data = []
    y_data = []
    for line in file:

        x = line.split()

        x_data.append([float(x[0]), float(x[1])])
        y_data.append(float(x[2]))

    file.close()
    print(x_data)
    print(y_data)
    return x_data, y_data


def check_validity_of_weights(x_data, y_data, object1):

    total = 0
    for i in range(len(y_data)):

        if object1.prediction(x_data[i]) == y_data[i]:
            total += 1

    accuracy_rate = total/len(y_data)*100

    print("\n")
    print("validity of weights = {}%".format(accuracy_rate))
    print("\n")


def main():

    x_training_data, y_training_data = read_file("/Users/masaaki/pythonPractice/training_data.txt")
    x_test_data, y_test_data = read_file("/Users/masaaki/pythonPractice/test_data.txt")

    object1 = Perceptron(0.5)
    check_validity_of_weights(x_test_data, y_test_data, object1)

    epoch = 30
    for i in range(epoch):
        for x, y in zip(x_training_data, y_training_data):
            object1.train(x, y)

        print(object1.weights, object1.bias)
    check_validity_of_weights(x_test_data, y_test_data, object1)


if __name__ == '__main__':

    main()






        

