import torch
import numpy as np
import copy

from torch.autograd import Variable


class DeepFoolAttack:
    def __init__(self, model, device, num_classes=10, overshoot=0.02, max_iter=8):
        self.model = model
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.device = device
    def deepfool(self, image):
        print(image)
        f_image = self.model(image)
        I = []
        for j in range(f_image.size(0)):
            tempI = (np.array(f_image[j].cpu().detach())).flatten().argsort()[::-1]
            I.append(tempI)
        I = np.array(I)


        for j in range(image.size(0)):
            imageToChange = copy.deepcopy(image[j].view(1, 1, 28, 28))
            w = np.zeros(image[j].shape)
            r_tot = np.zeros(image[j].shape)
            loopNumber = 0
            x = torch.tensor(imageToChange.to(self.device), requires_grad=True)
            fs = self.model(x)
            temp2I = I
            I = I[j]

            label = I[0]
            fs_list = [fs[0, I[k]] for k in range(self.num_classes)]
            newLabel = label

            while newLabel == label and loopNumber < self.max_iter:

                pert = np.inf  # the largest value
                fs[0, I[0]].backward(retain_graph=True)
                rightGradiant = x.grad.data.cpu().numpy().copy()

                for k in range(1, self.num_classes):

                    fs[0, I[k]].backward(retain_graph=True)
                    currentGradiant = x.grad.data.cpu().numpy().copy()

                    # set new w_k and new f_k
                    w_k = currentGradiant - rightGradiant  # calculating  value to be in other class
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                    tempPert = abs(f_k) / np.linalg.norm(
                        w_k.flatten())  # calculating the perturbations that we add to get to each class

                    # determine which w_k to use
                    if tempPert < pert:  # choose the minimum
                        pert = tempPert
                        w = w_k

                # compute r_i and r_tot
                # Added 1e-4 for numerical stability
                r_i = (pert + 1e-4) * w / np.linalg.norm(w)  # calculating the nearest hyperplane
                r_tot = np.float32(r_tot + r_i)
                imageToChange = image[j] + (1 + self.overshoot) * torch.from_numpy(
                    r_tot).cuda()  # adding the value to the image
                x = torch.tensor(imageToChange, requires_grad=True)
                fs = self.model(x)
                newLabel = np.argmax(fs.data.cpu().numpy().flatten())

                loopNumber += 1
            I = temp2I
            image[j] = imageToChange.view(1, 1, 28, 28)

        # print(image.shape)
        r_tot = (1 + self.overshoot) * r_tot
        return newLabel, (newLabel == label), image
    def evaluate_attack(self, test_dataloader, model):
        print("run")
        # print(len(self.test_dataloader))
        success_attacks = 0
        hit = 0

        for data, label in test_dataloader:
            # send data to device
            data, label = data.to(self.device), label.to(self.device)
            f_image = model(data)
            I = (np.array(f_image.cpu().detach())).flatten().argsort()[::-1]
            label_before = I[0]  # the label before
            if data is None:
                print("!!!!!!!")
                print(data)
                continue
            pert_label, isEqual, changedImage = self.deepfool(data)
            changedImage = changedImage.to(self.device)
            f_image = model(changedImage.cuda())
            I2 = (np.array(f_image.cpu().detach())).flatten().argsort()[::-1]
            label_after = I2[0]
            if label_before != label_after:
                success_attacks += 1
            if label_after != label:
                hit += 1

        print("Attack Success Rate = {} / {} = {}".format(success_attacks,
                                                          len(test_dataloader), success_attacks / len(test_dataloader)))
        print("Acuracy Rate = {} / {} = {}".format(hit,
                                                   len(test_dataloader), hit / len(test_dataloader)))
    def image_deepfool(self, image):
        f_image = self.model(image)
        I = (np.array(f_image.cpu().detach())).flatten().argsort()[::-1]
        label = I[0]

        imageToChange = copy.deepcopy(image)
        w = np.zeros(image.shape)
        r_tot = np.zeros(image.shape)

        loopNumber = 0

        x = torch.tensor(imageToChange.to(self.device), requires_grad=True)

        fs = self.model(x)
        fs_list = [fs[0, I[k]] for k in range(self.num_classes)]
        newLabel = label

        while newLabel == label and loopNumber < self.max_iter:

            pert = np.inf  # the largest value
            fs[0, I[0]].backward(retain_graph=True)
            rightGradiant = x.grad.data.cpu().numpy().copy()

            for k in range(1, self.num_classes):

                fs[0, I[k]].backward(retain_graph=True)
                currentGradiant = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = currentGradiant - rightGradiant  # calculating  value to be in other class
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                tempPert = abs(f_k) / np.linalg.norm(
                    w_k.flatten())  # calculating the perturbations that we add to get to each class

                # determine which w_k to use
                if tempPert < pert:  # choose the minimum
                    pert = tempPert
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)  # calculating the nearest hyperplane
            r_tot = np.float32(r_tot + r_i)

            imageToChange = image + (1 + self.overshoot) * torch.from_numpy(
                r_tot).cuda()  # adding the value to the image

            x = torch.tensor(imageToChange, requires_grad=True)
            fs = self.model(x)
            newLabel = np.argmax(fs.data.cpu().numpy().flatten())

            loopNumber += 1

        r_tot = (1 + self.overshoot) * r_tot

        return newLabel
    def return_noisy_batch(self, data):
            data = data.to(self.device)
            data = Variable(data.view(-1, 1, 28, 28))

            # shape = data.shape
            newLabel, isChanged, noisy_img = self.deepfool(data)
            return noisy_img
    def run(self, test_loader):
        # print(len(self.test_dataloader))
        success_attacks = 0
        for data, label in test_loader:
            # send dat to device
            data, label = data.to(self.device), label.to(self.device)
            pert_label = self.image_deepfool(data)
            if (pert_label.item() != label.item()):
                success_attacks += 1
        print("Attack Success Rate = {} ".format(success_attacks))




