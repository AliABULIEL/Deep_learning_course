import main
from torch.autograd import Variable
import torch
def calculate_acc(dataset_loader, model, batch_size):
    n_correct = 0
    n_total = 0
    for j, (data, label) in enumerate(dataset_loader):
        data, label = data.to(main.device), label.to(main.device)
        # flatten the image to vector of size 28*28
        data = Variable(data.view(batch_size, 1, 28, 28))
        # calculate output
        y_hat = model(data)
        # get the prediction
        predictions = torch.argmax(y_hat, dim=1)
        n_correct += torch.sum(predictions == label).type(torch.float32)
        n_total += data.shape[0]
    acc = (n_correct / n_total).item()
    return acc