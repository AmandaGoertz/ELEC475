#	python3 train.py -content_dir ./../../../datasets/COCO10k/ -style_dir ./../../../datasets/wikiart10k/
#	-gamma 1.0 -e 20 -b 32 -l encoder.pth -s decoder.pth -p decoder.png

import argparse
import os
import custom_dataset
from PIL import Image, ImageFile
import numpy
from torch.optim import Adam
from torch.optim import lr_scheduler
import AdaIN_net
import torchvision
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train(model, content_dataloader, style_dataloader, n_batch, optimizer, device):#n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):

    model.train()
    loss_c_array = []
    loss_s_array = []
    loss_array = []

    for epoch in range(0, args.n_epoch):
        loss_c_train = []
        loss_s_train = []
        loss_train = []
        loss_c_total = 0
        loss_s_total = 0
        loss_total = 0
        loss_avg = 0
        batch = 0
        print("epoch", epoch)

        for batch in range(0, n_batch):
        #for content_images, style_images in zip(content_dataloader, style_dataloader):
            print("batch", batch)
            #batch += 1
            content_images = next(iter(content_dataloader)).to(device=device)
            style_images = next(iter(style_dataloader)).to(device=device)

            loss_c, loss_s = model.forward(content_images, style_images)
            loss = loss_c + (args.g*loss_s)

            optimizer.zero_grad()  # reset optimizer gradients to zero
            loss.backward()
            optimizer.step()  # iterate the optimization, based on loss gradients

            loss_c_total += loss_c.item()
            loss_s_total += loss_s.item()
            loss_total += loss.item()

        # scheduler.step(loss_c_total)
        # scheduler.step(loss_s_total)
        # scheduler.step(loss_total)

        loss_c_train += [loss_c_total/len(content_dataloader.dataset)]
        loss_s_train += [loss_s_total/len(style_dataloader.dataset)]
        loss_train += [loss_total/len(content_dataloader.dataset)]

        loss_c_array.append(loss_c_train)
        loss_s_array.append(loss_s_train)
        loss_array.append(loss_train)

        torch.save(model.decoder.state_dict(), 'C:\\Users\\Amanda Goertz\\PycharmProjects\\Lab2\\decoder.pth')

        # loss = loss.item()
        # loss_avg += (loss/n_batch)
        # total_loss_array[epoch] = loss_avg

        #optimizer.zero_grad()

        # scheduler.step(loss_c_total)
        # scheduler.step(loss_s_total)
        print("content loss", loss_c_train)
        print("content loss array", loss_c_array)
        print("content loss", loss_s_train)
        print("content loss array", loss_s_array)
        print("content loss", loss_train)
        print("content loss array", loss_array)

        plt.plot(loss_c_array, label = 'Content Loss')
        plt.plot(loss_s_array, label = 'Style Loss')
        plt.plot(loss_array, label = 'Total Loss')
        plt.legend()
        plt.show()
        plt.savefig("LossPlot.png")

# <        f = plt.figure()
#
#         f.add_subplot(1, 3, 1)
#         plt.xlabel('epoch')
#         plt.ylabel('content loss')
#         plt.plot(epoch, loss_c_train)
#         f.add_subplot(1, 3, 2)
#         plt.plot(epoch, loss_s_train)
#         plt.xlabel('epoch')
#         plt.ylabel('style loss')
#         f.add_subplot(1, 3, 3)
#         plt.plot(epoch, loss_train)
#         plt.xlabel('epoch')
#         plt.ylabel('total loss')
#         plt.savefig('LossPlots.png')
#         plt.show()>


parser = argparse.ArgumentParser()

parser.add_argument('-c', '--content_dir')
parser.add_argument('-gamma', '--g', type=float)
parser.add_argument('-e', '--n_epoch', type=int)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('-l', '--encoder_path')
parser.add_argument('-s', '--decoder_path')
parser.add_argument('-p', '--decoder_png')
parser.add_argument('-u' '--cuda')

args = parser.parse_args()

decoder = AdaIN_net.encoder_decoder.decoder
encoder = AdaIN_net.encoder_decoder.encoder

# loading in weights
encoder.load_state_dict(torch.load(args.encoder_path))

model = AdaIN_net.AdaIN_net(encoder, decoder)

#custom_set = "./../../../datasets/COCO100/"
content_dir = "C:\\Users\\Amanda Goertz\\PycharmProjects\\Lab2\\COCO10k\\"
style_dir = "C:\\Users\\Amanda Goertz\\PycharmProjects\\Lab2\\wikiart10k\\"

#train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=512),
    torchvision.transforms.RandomCrop(256),
    torchvision.transforms.ToTensor()
])

content_set = custom_dataset.custom_dataset(content_dir, transform)
style_set = custom_dataset.custom_dataset(style_dir, transform)

n_batch = int(len(content_set) / args.batch_size)

#n_batch = args.batch_size
content_dataloader = torch.utils.data.DataLoader(content_set, args.batch_size, shuffle=True)
style_dataloader = torch.utils.data.DataLoader(style_set, args.batch_size, shuffle=True)

rate_learning = 1e-4
weight_decay = 1e-5
optimizer = Adam(model.decoder.parameters(), rate_learning, weight_decay=weight_decay)
#scheduler = lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=7, last_epoch=-1)
device = torch.device('cpu')

train(model, content_dataloader, style_dataloader, n_batch, optimizer, device)


