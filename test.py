import torch
from torch.autograd import Variable
import logging
from main import device
from utils import calculate_acc
logging.basicConfig(filename='test.log', encoding='utf-8')


def test_model(model, test_loader, epochs, loss_function,device):
    # evaluate model:
    model.eval()
    with torch.no_grad():
        print(" Model is {}".format(model))
        logging.info(" Model is {}".format(model))
        iteration = []
        test_accuracy = []
        count = 0
        for i in range(epochs):
            losses = []
            count += 1
            iteration.append(count)
            for j, (data, label) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                data = Variable(data.view(1, 1, 28, 28))
                label = Variable(label)
                y_hat = model(data)
                # calculate loss
                error = loss_function(y_hat, label)
                # backprop
                losses.append(error.detach())
            print("epoch {} | test loss : {} ".format(i, error.detach()))
            logging.info("epoch {} | test loss : {} ".format(i, error.detach()))
            test_acc = calculate_acc(test_loader, model, 1)
            test_accuracy.append(test_acc)

        print("test accuracy : {}".format(test_acc))
        logging.info("test accuracy : {}".format(test_acc))