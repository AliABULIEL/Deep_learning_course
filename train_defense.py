import torch
from torch.autograd import Variable
import utils
import matplotlib.pyplot as plt
import tfgsm_attack
import numpy as np

def train_model(model, train_loader, validation_loader, epochs, learning_rate, optimizer, loss_function, device, Y):
    print(" Model is {}".format(model))
    # logging.info(model)

    count = 0
    train_accuracy = []
    validate_accuracy = []
    iterations = []
    train_loss = []
    train_losses = []
    train_loss_attacked = []
    val_loss = []
    tfgsm = tfgsm_attack.FGSMAttack(model,[0.5],train_loader,device,Y.to(device))

    for i in range(epochs):
        correct = 0
        total = 0
        count += 1
        iterations.append(count)
        print("*******************************")
        for j, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            data = Variable(data.view(100, 1, 28, 28))
            label = Variable(label)
            optimizer.zero_grad()
            # flatten the image to vector of size 28*28
            # data = data.view(-1, n_features)
            # calculate output
            data.requires_grad = True
            y_hat = model(data)
            # calculate loss
            error = loss_function(y_hat, label)
            error.backward()
            attacked_data = tfgsm.perturb_image(data, 0.5, data.grad)
            train_losses.append(error.data.item())
            # backprop
            optimizer.step()
            train_losses.append(error.detach().cpu().numpy())
            # train the attacked data


            optimizer.zero_grad()
            y_attacked = model(attacked_data.to(device))
            error_attacked = loss_function(y_attacked, label)
            error_attacked.backward()
            optimizer.step()
            train_loss_attacked.append(error_attacked.detach().cpu().numpy())
            preds_np = y_attacked.cpu().detach().numpy()
            correct += (np.argmax(preds_np, axis=1) == label.cpu().detach().numpy()).sum()
            total = total+train_loader.batch_size
        avg = np.mean(train_losses)
        train_losses = []
        print("epoch {} | train loss : {} ".format(i, avg))
        train_loss.append(avg)
        avg = np.mean(train_loss_attacked)
        train_loss_attacked = []
        print("epoch {} | train loss after attack  : {} ".format(i, avg))
        print("Successful attack  {} , correct predctions for this epoch {} / {}".format(total-correct, correct, total))
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
                # backprop
                val_lossess.append(error.detach().cpu().numpy())
        avg = np.mean(val_lossess)
        val_lossess = []
        print("epoch {} | Validation loss : {} ".format(i, avg))
        val_loss.append(avg)
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
    plt.title("Iterations vs Loss function")
    plt.show()

    plt.plot(iterations, val_loss)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Valdiation loss")
    plt.title("Iterations vs Loss function")
    plt.show()

    return model

