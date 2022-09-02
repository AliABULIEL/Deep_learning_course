import argparse
import data_set
import model
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--train", nargs='?', help="train NN for FashionMinst dataset", const=True)
parser.add_argument("-e", "--test", nargs='?', help="test NN for FashionMinst dataset", const=True)
parser.add_argument("-f", "--TFGSM", nargs='?',  help="run TFGSM adversial attack", const=True)
parser.add_argument("-d", "--DEEPFOOL", nargs='?',  help="run Deep fool adversial attack", const=True)
parser.add_argument("-s", "--defense", nargs='?',  help="run Deep fool adversial attack", const=True)


args = parser.parse_args()

trained_model= None
train_loader, val_loader, test_loader = data_set.load_dataset()
fashion_model = model.Fashion_MNIST_CNN()
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
fashion_model.to(device)
X, Y = next(iter(test_loader))

if args.train:
    import train
    trained_model = train.train_model(model=fashion_model, train_loader=train_loader, validation_loader=val_loader, epochs=20, learning_rate=0.001, optimizer=torch.optim.Adam(fashion_model.parameters(), lr=0.001), loss_function=nn.CrossEntropyLoss(), device=device, patience=5)
if args.test:
    import test
    test.test_model(model=trained_model, test_loader=test_loader, epochs=1, loss_function=nn.CrossEntropyLoss(), device=device)
if args.TFGSM:
    import tfgsm_attack
    attack = tfgsm_attack.FGSMAttack(trained_model,[0.5],test_loader,device,Y.to(device))
    attack.run()
if args.DEEPFOOL:
    import deep_fool
    deep_fool_instance = deep_fool.DeepFoolAttack(trained_model, test_loader,device)
    deep_fool_instance.run()
if args.defense:
    import train_defense
    import tfgsm_attack
    fashion_model_defensed = model.Fashion_MNIST_CNN()
    trained_model_defensed = train_defense.train_model(model=fashion_model, train_loader=train_loader, validation_loader=val_loader, epochs=7, learning_rate=0.001, optimizer=torch.optim.Adam(fashion_model.parameters(), lr=0.001), loss_function=nn.CrossEntropyLoss(), device=device, Y=Y.to(device))
    attack = tfgsm_attack.FGSMAttack(trained_model_defensed, [0.5], test_loader, device, Y.to(device))
    attack.run()


