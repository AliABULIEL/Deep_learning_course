import torch
from torch import nn
import matplotlib.pyplot as plt


class FGSMAttack():
    def __init__(self, model, epsilons, test_dataloader, device, target):
        self.model = model
        self.epsilons = epsilons
        self.test_dataloader = test_dataloader
        self.target = target
        self.adv_examples = {}
        self.device = device

    def perturb_image(self, x, eps, grad):
        x_prime = x - eps * grad.sign()
        # keep image data in the [0,1] range
        x_prime = torch.clamp(x_prime, 0, 1)
        show = plt.imshow(x_prime.detahc().cpu().numpy())
        plt.show()
        return x_prime

    def run(self):
        # run the attack for each epsilon
        self.model.to(self.device)
        for epsReal in self.epsilons:
            self.adv_examples[epsReal] = []  # store some adv samples for visualization
            eps = epsReal - 1e-7  # small constant to offset floating-point errors
            successful_attacks = 0
            for data, label in self.test_dataloader:
                # send dat to device
                data, label = data.to(self.device), label.to(self.device)

                # TFGSM attack requires gradients
                data.requires_grad = True

                output = self.model(data)
                init_pred = output.argmax(dim=1, keepdim=True)
                if init_pred.item() != label.item():
                    # image is not correctly predicted to begin with, skip
                    continue
                if self.target and self.target == label.item():
                    # if the image has the target class, skip
                    continue

                # calculate the loss
                L = nn.CrossEntropyLoss()
                prediction = torch.argmax(output, dim=1)
                loss = L(output, self.target)

                # zero out all existing gradients
                self.model.zero_grad()
                # calculate gradients
                loss.backward()
                data_grad = data.grad

                perturbed_data = self.perturb_image(data, eps, data_grad)

                # predict class for adversarial sample
                adv_output = self.model(perturbed_data)
                adv_pred = adv_output.argmax(dim=1, keepdim=True)
                if adv_pred.item() == self.target:
                    successful_attacks += 1
                    if len(self.adv_examples[epsReal]) < 5:
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        self.adv_examples[epsReal].append((init_pred.item(), adv_pred.item(), adv_ex))
                        # print status line
            success_rate = successful_attacks / float(len(self.test_dataloader))
            print("Epsilon: {}\tAttack Success Rate = {} / {} = {}".format(epsReal, successful_attacks,
                                                                           len(self.test_dataloader), success_rate))
            # logging.info("Epsilon: {}\tAttack Success Rate = {} / {} = {}".format(epsReal, successful_attacks,
            #                                                                len(self.test_dataloader), success_rate))