from torch.optim import SGD
import torch.nn as nn
import numpy as np

class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.optimizer = SGD(self.model.parameters(), lr=1e-3)

def forward(model, data):
    return model.model(data)
    
def backward(model, input, labels, loss):
    model.optimizer.zero_grad()
    loss.backward()

def gradient_descent_step(model, grad, learning_rate):
    model.optimizer.step()
    return model

criterion = nn.MSELoss()
def compute_loss(outputs, labels):
    return criterion(outputs, labels)

def check_answer(x, y):
    implementation_error = np.mean((x - y) ** 2)

    print(f"Difference between correct answer and your solution is {implementation_error}")
    if implementation_error < 1e-10:
        print("Passed!")
    else:
        print("Not quite..")

