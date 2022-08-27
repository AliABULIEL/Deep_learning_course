import argparse
import train
import data_set
import model
import test
import tfgsm_attack
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()

parser.add_argument("-t", "--train", nargs='?', help="train NN for FashionMinst dataset", const=True)
parser.add_argument("-e", "--test", nargs='?', help="test NN for FashionMinst dataset", const=True)
parser.add_argument("-f", "--TFGSM", nargs='?',  help="run TFGSM adversial attack", const=True)

args = parser.parse_args()
global train_loader
global val_loader
global test_loader
global fashion_model
global train
global trained_model
global test
global tfgsm_attack
global device
trained_model= None
train_loader, val_loader , test_loader = data_set.load_dataset()
fashion_model = model.Fashion_MNIST_CNN()
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
fashion_model.to(device)

if(args.train):
    iterations = []
    train_accuracy = []
    validate_accuracy = []
    trained_model= train.train_model(model=fashion_model, train_loader=train_loader, validation_loader=val_loader, epochs=12, learning_rate=0.001, optimizer=torch.optim.Adam(fashion_model.parameters(), lr=0.001), loss_function=nn.CrossEntropyLoss(), device=device, train_accuracy=train_accuracy, iterations=iterations, validate_accuracy=validate_accuracy)

    plt.plot(iterations, train_accuracy)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Train Accuracy")
    plt.title("Iterations vs Accuracy")
    plt.show()

    plt.plot(iterations, validate_accuracy)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Valdiation Accuracy")
    plt.title("Iterations vs Accuracy")
    plt.show()
if(args.test):
    test.test_model(model=trained_model,test_loader=test_loader,epochs=1,loss_function=nn.CrossEntropyLoss(),device=device)
if(args.TFGSM):
    X, Y = next(iter(test_loader))
    attack = tfgsm_attack.FGSMAttack(trained_model,[0.5],test_loader,device,Y.to(device))
    attack.run()

