import numpy as np
import matplotlib.pyplot as plt


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        
        if 0.1 * i == 0.5:
            continue
        
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')  # 紅色表示類別 0
        else:
            plt.plot(x[i][0], x[i][1], 'bo')  # 藍色表示類別 1

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')  # 紅色表示類別 0
        else:
            plt.plot(x[i][0], x[i][1], 'bo')  # 藍色表示類別 1

    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return np.multiply(s, 1.0 - s)


class model:
    
    n_lin1 = 3
    n_lin2 = 4
    
    def __init__(self, train_X, train_label):
        self.train_X = train_X
        self.train_label = train_label
        self.W1 = np.random.uniform(low=0, high=1.0, size=(train_X.shape[1], self.n_lin1))
        self.W2 = np.random.uniform(low=0, high=1.0, size=(self.n_lin1, self.n_lin2))
        self.W3 = np.random.uniform(low=0, high=1.0, size=(self.n_lin2, 1))

    def train(self, lr, n_epochs):
        for epoch in range(n_epochs):
            # forward
            H1 = np.dot(self.train_X, self.W1)
            Z1 = sigmoid(H1)
            H2 = np.dot(Z1, self.W2)
            Z2 = sigmoid(H2)
            H3 = np.dot(Z2, self.W3)
            y = sigmoid(H3)
            
            
            loss = np.mean((y - self.train_label)**2)
            print(f"epoch {epoch}: {loss}")
            
            #  backward            
            dy = 2 * (y - self.train_label) / self.train_X.shape[0] # derivative of MSE: 1/n * sum((y - y_hat)^2)
            # dH3 = np.dot(sigmoid_derivative(H3).T, dy)
            dH3 = dy * sigmoid_derivative(y)
            # print(dy.shape)
            # print(dH3.shape)
            # print(Z2.shape)
            dW3 = np.dot(Z2.T, dH3)
            
            dZ2 = np.dot(dH3, self.W3.T)
            # dH2 = np.dot(sigmoid_derivative(H2).T, dZ2)
            dH2 = dZ2 * sigmoid_derivative(Z2)
            dW2 = np.dot(Z1.T, dH2)

            dZ1 = np.dot(dH2, self.W2.T)
            # dH1 = np.dot(sigmoid_derivative(H1).T, dZ1)
            dH1 = dZ1 * sigmoid_derivative(Z1)
            dW1 = np.dot(self.train_X.T, dH1)
            
            self.W1 -= lr * dW1
            self.W2 -= lr * dW2
            self.W3 -= lr * dW3
            
            # self.W1 += lr * dW1
            # self.W2 += lr * dW2
            # self.W3 += lr * dW3

    def predict(self, test_X, test_label):
        H1 = np.dot(test_X, self.W1)
        Z1 = sigmoid(H1)
        H2 = np.dot(Z1, self.W2)
        Z2 = sigmoid(H2)
        H3 = np.dot(Z2, self.W3)
        y = sigmoid(H3)

        loss = np.mean((y-test_label)**2)
        
        for i in range(test_X.shape[0]):
            print(f"iter {i} | ground truth: {test_label[i]} | pred: {y[i]}")
        
        
        return loss



def liner():
    lin_X, lin_label = generate_linear()
    lin_model = model(lin_X, lin_label)
    lin_model.train(lr=0.1, n_epochs=100000)
    lin_model.predict(lin_X, lin_label)
        

def xor():
    xor_X, xor_label = generate_XOR_easy()
    xor_model = model(xor_X, xor_label)
    xor_model.train(lr=0.1, n_epochs=100000)
    xor_model.predict(xor_X, xor_label)



if __name__ == "__main__":
    liner()
    # xor()
