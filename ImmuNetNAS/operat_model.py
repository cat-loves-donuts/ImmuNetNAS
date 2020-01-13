import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision

def train_model(model, linear, input_CNN, learning_rate, nsamples, load_train_dataset):
    model.train(True)
    linear.train(True)
    input_CNN.train(True)
    epoch = 0
    print_loss = 0

    "Setting the optimizer parameters"
    criterion = nn.CrossEntropyLoss()
    optimizer_module = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer_linear = optim.SGD(linear.parameters(), lr=learning_rate)
    optimizer_input_CNN = optim.SGD(input_CNN.parameters(), lr=learning_rate)

    "Check which divices is working and push the data into the divice"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    linear = linear.to(device)
    input_CNN = input_CNN.to(device)

    "train the model"
    for i, (data, target) in enumerate(load_train_dataset):
        data = data.to(device)
        target = target.to(device)

        if i > nsamples:
            break
        else:
            # train network
            out = input_CNN(data)
            out = model(out)
            out = linear(out)
            loss = criterion(out, target)
            print_loss = loss.data.item()

            # update the parameters
            optimizer_module.zero_grad()
            optimizer_linear.zero_grad()
            optimizer_input_CNN.zero_grad()
            loss.backward()
            optimizer_module.step()
            optimizer_linear.step()
            optimizer_input_CNN.step()
            epoch += 1
            if epoch % 390 == 0:
                print(' Train epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
    return print_loss


def test_model(model, linear, input_CNN, load_test_dataset,test_dataset, nsamples):
    model.eval()
    linear.eval()
    input_CNN.eval()
    eval_loss = 0
    eval_acc = 0
    criterion = nn.CrossEntropyLoss()

    "Check which divices is working and push the data into the divice"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    linear = linear.to(device)
    input_CNN = input_CNN.to(device)

    "test the model"
    for i, (data, target) in enumerate(load_test_dataset):
        data = data.to(device)
        target = target.to(device)
    #     if i > nsamples:
    #         break
    #     else:
        # get images, labels
        out = input_CNN(data)
        out = model(out)
        out = linear(out)
        loss = criterion(out, target)
        print_loss = loss.data.item()

            # loss funtion calculation
        eval_loss += loss.data.item() * target.size(0)
            # get the biggest predicting result in 10 classes
        _, pred = torch.max(out, 1)
        #print("Predict result:",pred,"the target result:",target)
            # does the prediction = the lable
        num_correct = (pred == target).sum()

            # get the amount of right predicting
        eval_acc += num_correct.item()

        # print the results
    print('\nTest Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))
    return eval_acc / (len(test_dataset))




