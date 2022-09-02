import torch
from torch.autograd import Variable
import utils
import matplotlib.pyplot as plt
import numpy as np



def train_model(model, train_loader, validation_loader, epochs, learning_rate, optimizer, loss_function, device):
    print(" Model is {}".format(model))
    # logging.info(model)
    count = 0
    train_accuracy = []
    validate_accuracy = []
    iterations = []
    train_loss = []
    val_loss = []
    train_losses = []
    best_loss = 100
    count = 0
    patience = 5
    best_model = None
    for i in range(epochs):
        count += 1

        print("*******************************")
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
            train_losses.append(error.detach().cpu().numpy())
        avg = np.mean(train_losses)
        train_losses = []
        print("epoch {} | train loss : {} ".format(i, avg))
        train_loss.append(avg)
        model.eval()

        with torch.no_grad():
            val_lossess = []
            for j, (data, label) in enumerate(validation_loader):
                data, label = data.to(device), label.to(device)
                data = Variable(data.view(100, 1, 28, 28))
                label = Variable(label)
                # flatten the image to vector of size 28*28
                y_hat = model(data)
                # calculate loss
                error = loss_function(y_hat, label)
                val_lossess.append(error.detach().cpu().numpy())
            current_loss = np.average(val_lossess)
            # print(" best loss is " + str(best_loss))
            # print(" current loss is " + str(current_loss))
            # print("patience 1 is " + str(patience))
            if(current_loss< best_loss):
                    print("patience reset")
                    best_loss = np.average(val_lossess)
                    torch.save(model.state_dict(), F"/content/gdrive/My Drive/best_model.pt")
                    patience = 5
                    best_model = model
            elif(current_loss >= best_loss):
                    # print("patince decrease " + str(patience))
                    patience -= 1

            if(patience == 0):
                    print("early stopping")
                    break



                # backprop
        iterations.append(count)
        avg = np.mean(val_lossess)
        val_lossess = []
        print("epoch {} | Validation loss : {} ".format(i,avg))
        val_loss.append(avg)
        # logging.info("epoch {} | train loss : {} ".format(i, error.detach()))
        train_acc = utils.calculate_acc(train_loader, model, 100,device)
        val_acc = utils.calculate_acc(validation_loader, model, 100,device)
        train_accuracy.append(train_acc)
        validate_accuracy.append(val_acc)

    # plot accuarcy
    # logging.info("train accuracy : {}".format(train_acc))
    # logging.info("Validation accuracy : {}".format(val_acc))
    print("last iteration " + str(iterations[-1]))
    print(" whre is patience" + str(iterations[-1]-patience))
    print(len(iterations))
    print(len(train_loss))
    print(len(val_loss))
    print("train accuracy : {}".format(train_acc))
    print("Validation accuracy : {}".format(val_acc))
    plt.plot(iterations, train_accuracy)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Train Accuracy")
    plt.title("Iterations vs Accuracy")
    plt.axvline(x=iterations[-1]-patience, color='b', ls='--')
    plt.show()

    plt.plot(iterations, validate_accuracy)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Valdiation Accuracy")
    plt.title("Iterations vs Accuracy")
    plt.axvline(x=iterations[-1]-patience, color='b', ls='--')
    plt.show()

    plt.plot(iterations[:-2], train_loss)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Train loss")
    plt.title("Iterations vs Loss function")
    plt.axvline(x=iterations[-1]-patience, color='b', ls='--')
    plt.show()

    plt.plot(iterations, val_loss)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Valdiation loss")
    plt.title("Iterations vs Loss function")
    plt.axvline(x=iterations[-1]-patience, color='b', ls='--')
    plt.show()

    return best_model

