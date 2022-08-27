from torch.autograd import Variable
import utils
import os



def train_model(model, train_loader, validation_loader, epochs, learning_rate, optimizer, loss_function,device,train_accuracy,validate_accuracy,iterations):
    print(" Model is {}".format(model))
    # logging.info(model)

    count = 0
    for i in range(epochs):
        losses = []
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
            losses.append(error.detach())

        print("epoch {} | train loss : {} ".format(i, error.detach()))
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

    return model

