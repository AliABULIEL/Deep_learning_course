import torch
from torch.autograd import Variable
import utils
import matplotlib.pyplot as plt


def train_model(model, train_loader, validation_loader, epochs, learning_rate, optimizer, loss_function, device):
    print(" Model is {}".format(model))
    # logging.info(model)

    count = 0
    train_accuracy = []
    validate_accuracy = []
    iterations = []
    train_loss = []
    val_loss = []
    for i in range(epochs):
        count += 1
        iterations.append(count)
        for j, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            data = Variable(data.view(100, 1, 28, 28))
            label = Variable(label)
            optimizer.zero_grad()
            # flatten the image to vector of size 28*28
            # data = data.view(-1, n_features)
            # calculate output
            y_hat = model(data)
            # calculate loss
            error = loss_function(y_hat, label)
            error.backward()
            # backprop
            optimizer.step()
        print("epoch {} | train loss : {} ".format(i, error.detach()))
        train_loss.append(error.detach().to("cpu").numpy())
        model.eval()
        with torch.no_grad():
            for j, (data, label) in enumerate(validation_loader):
                data, label = data.to(device), label.to(device)
                data = Variable(data.view(100, 1, 28, 28))
                label = Variable(label)
                # flatten the image to vector of size 28*28
                y_hat = model(data)
                # calculate loss
                error = loss_function(y_hat, label)
                # backprop

        print("epoch {} | Validation loss : {} ".format(i, error.detach()))
        print("\n")
        val_loss.append(error.detach().to("cpu").numpy())
        # logging.info("epoch {} | train loss : {} ".format(i, error.detach()))
        train_acc = utils.calculate_acc(train_loader, model, 100,device)
        val_acc = utils.calculate_acc(validation_loader, model, 100,device)
        train_accuracy.append(train_acc)
        validate_accuracy.append(val_acc)

    # plot accuarcy
    # logging.info("train accuracy : {}".format(train_acc))
    # logging.info("Validation accuracy : {}".format(val_acc))
    print("train accuracy : {}".format(train_acc))
    print("Validation accuracy : {}".format(val_acc))
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

    plt.plot(iterations, train_loss)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Train loss")
    plt.title("Iterations vs Accuracy")
    plt.show()

    plt.plot(iterations, val_loss)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Valdiation loss")
    plt.title("Iterations vs Accuracy")
    plt.show()

    return model

