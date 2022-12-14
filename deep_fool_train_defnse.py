import torch
from torch.autograd import Variable
import utils
import matplotlib.pyplot as plt
import deep_fool
import numpy as np


def train_model(model, train_loader, validation_loader, epochs, learning_rate, optimizer, loss_function, device, patience):
    print(" Model is {}".format(model))
    # logging.info(model)

    count = 0
    train_accuracy = []
    validate_accuracy = []
    iterations = []
    train_loss = []
    train_losses = []
    train_loss_attack = []
    train_loss_attacked = []
    val_loss = []
    attack_model = deep_fool.DeepFoolAttack(model=model, device=device,max_iter=4)
    best_loss = 100
    count = 0
    temp_patience = patience
    best_model = None
    model.to(device)

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
            attacked_data = attack_model.return_noisy_batch(data)
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
        print("epoch "+str(i)+" | Successful attack " + str(total-correct)+"  , correct predctions for this epoch "+str(correct)+ " / "+ str(total))
        train_loss_attack.append(avg)
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
            current_loss = np.mean(val_lossess)
            # print(" temp patience is "+ str(temp_patience))
            # print(" current loss is "+ str(current_loss))
            # print(" best loss is " + str(best_loss))
            if (current_loss < best_loss):
                # print("patience reset")
                torch.save(model.state_dict(), F"/content/gdrive/My Drive/best_model.pt")
                temp_patience = patience
                best_model = model
                best_loss = current_loss
            elif (current_loss >= best_loss):

                temp_patience -= 1
                # print("patince decrease " + str(temp_patience))

            if (temp_patience == 0):
                print("early stopping")
                break
        avg = np.mean(val_lossess)
        val_lossess = []
        print("epoch {} | Validation loss : {} ".format(i, avg))
        val_loss.append(avg)
        # logging.info("epoch {} | train loss : {} ".format(i, error.detach()))
        train_acc = utils.calculate_acc(train_loader, model, 100,device)
        val_acc = utils.calculate_acc(validation_loader, model, 100,device)
        train_accuracy.append(train_acc)
        validate_accuracy.append(val_acc)


    print("train accuracy : {}".format(train_accuracy[-patience]))
    print("Validation accuracy : {}".format(validate_accuracy[-patience]))



    plt.plot(iterations[:-1], validate_accuracy)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Valdiation Accuracy")
    plt.title("Iterations vs Accuracy")
    plt.axvline(x=iterations[-1] - patience , color='b', ls='--')

    plt.show()
    plt.plot(iterations[:-1], val_loss)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Valdiation loss")
    plt.title("Iterations vs Loss function")
    plt.axvline(x=iterations[-1] - patience , color='b', ls='--')
    plt.show()

    plt.plot(iterations, train_loss_attack)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Train loss after attack and defense")
    plt.title("Iterations vs Loss function")
    plt.axvline(x=iterations[-1] - patience , color='b', ls='--')
    plt.show()

    # x = iterations[-1]
    # x += 1
    # iterations.append(x)
    plt.plot(iterations, train_loss)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Train loss")
    plt.title("Iterations vs Loss function")
    plt.axvline(x=iterations[-1] - patience , color='b', ls='--')
    plt.show()
    plt.plot(iterations[:-1], train_accuracy)
    plt.xlabel("No. of Iteration")
    plt.ylabel(" Train Accuracy")
    plt.title("Iterations vs Accuracy")
    plt.axvline(x=iterations[-1] - patience , color='b', ls='--')
    plt.show()


    return best_model

