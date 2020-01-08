# !/usr/bin/python
# coding: utf8
from __future__ import print_function, division

import random
import torch
import numpy as np
import torchvision
import time
import os
import copy
from mutation import structure_mutate
from torchvision import transforms
from delete import delete_file, delete_all
from input_CNN_part import Input_CNN_block
from Final import CNN_final_block
from Linear_part import Linear_block
from cifr10_test import train_model, test_model
from connect import connection

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 16

def load_train_dataset():
    data_path = 'C:\\Users\\garychenai\\Desktop\\Final project\\codes\\CIFR-10\\train'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def load_test_dataset():
    data_path = 'C:\\Users\\garychenai\\Desktop\\Final project\\codes\\CIFR-10\\test'
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=0,
        shuffle=True
    )
    return test_loader

def test_dataset():
    data_path = 'C:\\Users\\garychenai\\Desktop\\Final project\\codes\\CIFR-10\\test'
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    return test_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

STAGES = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])  # stages
NUM_NODES = np.array([7, 7, 7, 7, 7, 7])  # K in each stage
generation = 20
#nsamples = int(50000/batch_size)/(len(NUM_NODES))
nsamples = 1
learning_rate = 1e-3
highest_clone = []
highest_clone_fitness = []
highest_clone_index = []
highest_model_structure = []
saved_structure = []
check_model = []
check_model_structure = []
check_model_result = []
remainDirsList = ['log.txt']
tracker = 0
n_epochs = 1000
population = 50  # populations of each stage

cuda_gpu = torch.cuda.is_available()

if (cuda_gpu):

    for m in range(0, len(NUM_NODES)):
        initial_model = [[0] * 2 for row in range(population)]  # initialize the models
        model_results_fitness = [[0] * 2 for row in range(population)]
        model_clone = []
        model_index = []
        x = []
        for i in range(0, population):
            model_index.append(i+1)

        print("stage", (m + 1), " have ", NUM_NODES[m], "layers")
        print("start to initialzation the population: ")

        time_checker = 0

        while time_checker < population:
            checksructure = 0
            for j in range(0, NUM_NODES[m]):
                x.append(random.choice([1, 2, 3, 4, 5, 6, 7, 8]))
            for q in range(0, NUM_NODES[m]):
                if x[q] == 5 or x[q] == 6 or x[q] == 7 or x[q] == 8:
                    checksructure = checksructure + 1
            if checksructure == NUM_NODES[m]:
                x = []
            else:
                initial_model[time_checker][0] = x
                x = []
                for k in range(0, int(NUM_NODES[m] * (NUM_NODES[m] + 1) / 2)):
                    x.append(random.choice([0, 1]))
                initial_model[time_checker][1] = x
                x = []
                time_checker = time_checker + 1


        if tracker > 0:
            current_model = []
            lenth = 0
            # for c in range(0, tracker):
            #     lenth = (1 + NUM_NODES[c]) * NUM_NODES[c] / 2
            #     current_model.append(CNN_final_block(saved_structure[0][0:NUM_NODES[c]], NUM_NODES[c],
            #                                          saved_structure[1][0:int(lenth)], 64, 64))
            #     saved_structure[0] = saved_structure[0][NUM_NODES[c]:]
            #     saved_structure[1] = saved_structure[1][int(lenth):]
            #
            # higest_model = current_model[0]
            # current_model = current_model[1:]
            # for c in range(0, tracker - 1):
            #     higest_model = connection(higest_model, current_model[c])
            # print("highest model", higest_model)
            higest_model = torch.load("C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")
            for i in range(0, population):
                input_CNN = Input_CNN_block(3, 64)
                model2 = CNN_final_block(initial_model[i][0], NUM_NODES[m], initial_model[i][1], 64, 64)
                model = connection(higest_model, model2)
                linear = Linear_block(65536, 10)
                model_results_fitness[i][0] = train_model(model, linear, input_CNN, learning_rate, nsamples,
                                                          load_train_dataset())  # save training loss in [0]
                torch.save(input_CNN,
                           "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\input_CNN_s%d_module_%d.pth" % (
                           tracker + 1, i+1))
                torch.save(linear,
                           "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\linear_s%d_module_%d.pth" % (
                           tracker + 1, i+1))
                torch.save(model,
                           "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\model_s%d_module_%d.pth" % (
                               tracker + 1, i + 1))

                model_results_fitness[i][1] = test_model(model, linear, input_CNN, load_test_dataset(), test_dataset(),
                                                         nsamples)  # save test accuracy in [1]
        else:
            for i in range(0, population):
                input_CNN = Input_CNN_block(3, 64)
                model = CNN_final_block(initial_model[i][0], NUM_NODES[m], initial_model[i][1], 64, 64)
                linear = Linear_block(65536, 10)
                print("model structure : ", initial_model[i])
                model_results_fitness[i][0] = train_model(model, linear, input_CNN, learning_rate, nsamples,
                                                          load_train_dataset())  # save training loss in [0]
                torch.save(input_CNN,
                           "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\input_CNN_s%d_module_%d.pth"%(tracker+1, i+1))
                torch.save(linear,
                           "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\linear_s%d_module_%d.pth"%(tracker+1, i+1))
                torch.save(model,
                           "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\model_s%d_module_%d.pth" % (
                               tracker + 1, i + 1))
                model_results_fitness[i][1] = test_model(model, linear, input_CNN, load_test_dataset(), test_dataset(),
                                                         nsamples)  # save test accuracy in [1]

        print("clone the population to clone pool")
        model_clone = copy.deepcopy(initial_model)  # clone initial models for immune operations
        print("start to keep upload the generations")

        for v in range(0, generation):
            highest_clone = []
            highest_clone_fitness = []
            highest_clone_index = []
            delete = []
            delete1 = []
            delete_num = []
            delete_num1 = []
            temp_del = []
            print("this is the ", v + 1, "th generation")
            print("delete the same structrue models")


            print("index before delete: ",model_index )
            print("fitneess of models before delete: ", model_results_fitness)
            for r1 in range(0, len(model_clone) - 1):
                for r2 in range(r1 + 1, len(model_clone)):
                    checker = 0
                    point = 0
                    for r3 in range(0, len(model_clone[0][0])):
                        if model_clone[r1][1][-(r3 + 1)] == model_clone[r2][1][-(r3 + 1)]:
                            if model_clone[r1][1][-(r3 + 1)] == 1:
                                checker = checker + 1
                                if model_clone[r1][0][-(r3 + 1)] == model_clone[r2][0][-(r3 + 1)]:
                                    point = point + 1
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass
                    num_one = 0
                    for r4 in range(0, len(model_clone[r1][0])):
                        if model_clone[r1][0][r4] == 1:
                            num_one = num_one + 1
                    if checker == num_one and point == num_one:
                        if model_results_fitness[r1] > model_results_fitness[r2]:
                            delete1.append(model_index[r2])
                            delete_num1.append(r2)
                        elif model_results_fitness[r1] < model_results_fitness[r2]:
                            delete1.append(model_index[r1])
                            delete_num1.append(r1)
                        else:
                            pass
                    else:
                        pass
            delete = list(set(delete1))
            delete.sort(key=delete1.index)
            delete_num = list(set(delete_num1))
            delete_num.sort(key=delete_num1.index)
            print("delet num after ordering: ", delete_num)
            print("delet after ordering: ", delete)
            print("model index: ", model_index)

            for i in range(0, len(delete_num)):
                for j in range(0, len(delete_num) - 1 - i):
                    if delete_num[j] > delete_num[j + 1]:
                        temp_del = delete_num[j]
                        delete_num[j] = delete_num[j + 1]
                        delete_num[j + 1] = temp_del

                        temp_del = delete[j]
                        delete[j] = delete[j + 1]
                        delete[j + 1] = temp_del
            print("delet num : ",delete_num)
            print("delet: ",delete)

            if len(delete) >= 5:
                for r5 in range(0, 5):
                    delete_file(int(delete[r5]), tracker + 1)
                    model_index.remove(delete[r5])
                    model_clone.pop((delete_num[r5]-r5))
                    model_results_fitness.pop((delete_num[r5]-r5))
            else:
                for r6 in range(0, len(delete)):
                    delete_file(int(delete[r6]), tracker + 1)
                    model_index.remove(delete[r6])
                    model_clone.pop((delete_num[r6]-r6))
                    model_results_fitness.pop((delete_num[r6]-r6))
                for r7 in range(0, 5 - len(delete)):
                    delete_file(model_index[0], tracker + 1)
                    model_index.pop(0)
                    model_clone.pop(0)
                    model_results_fitness.pop(0)



            for i in range(0, len(model_clone)):  # ordering the clone by fitness(accuracy) to
                for j in range(0, len(model_clone) - 1 - i):  # select the highest 25% in to
                    if model_results_fitness[j][1] > model_results_fitness[j + 1][1]:
                        temp_oder = model_results_fitness[j]
                        model_results_fitness[j] = model_results_fitness[j + 1]
                        model_results_fitness[j + 1] = temp_oder

                        temp_oder = model_clone[j]
                        model_clone[j] = model_clone[j + 1]
                        model_clone[j + 1] = temp_oder

                        temp_oder = model_index[j]
                        model_index[j] = model_index[j + 1]
                        model_index[j +1] = temp_oder
            print("In ", v + 1, "th genetation, get the highest clones")
            for i in range(0, int(len(model_clone) * 0.25)+1):  # select highest 25% clone and save them
                highest_clone.append(model_clone[-(i + 1)])
                highest_clone_fitness.append(model_results_fitness[-(i + 1)])
                highest_clone_index.append(model_index[-(i + 1)])
            print("highest clones:", (int(len(model_clone) * 0.25)+1), "clone numbers: ", len(model_clone))
            print("In ", v + 1, "th genetation, mutate highest clones")


            time_checker = 0
            # temp_model = [[0] * 2 for row in range(len(highest_clone))]
            while time_checker < len(highest_clone):  # mutate the highest clone pool by affinity
                y = []  # calculate affnity and add in to the arry
                checksructure = 0
                pointstructure = 0

                y.append(structure_mutate(highest_clone,highest_clone_fitness,len(highest_clone),population,time_checker))##########更改为mutate_structure

                for q in range(0, NUM_NODES[m]):
                    if y[0][1][-(q + 1)] == 1:
                        checksructure = checksructure + 1
                        if y[0][0][-(q+1)] == 5 or y[0][0][-(q+1)] == 6 or y[0][0][-(q+1)] == 7 or y[0][0][-(q+1)] == 8 :
                            pointstructure = pointstructure + 1
                        else:
                            pass
                    else:
                        pass

                if checksructure == pointstructure:
                    y = []
                else:
                    # temp_model[time_checker] = y
                    # y = []
                    # for k in range(0, int(NUM_NODES[m] * (NUM_NODES[m] + 1) / 2)):
                    #     y.append(random.choice([0, 1]))
                    # temp_model[time_checker][1] = y
                    model_clone.append(y[0])
                    time_checker  = time_checker + 1
                    model_index.append((population + int((population*0.25))*v + 5*v + time_checker))############################
                    # print("new added model: ",model_clone[-1])
                    # print("number of population:",len(model_clone))
                    if tracker > 0:
                        current_model = []
                        input_CNN = Input_CNN_block(3, 64)
                        model2 = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
                        higest_model = torch.load("C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")
                        model = connection(higest_model, model2)
                        linear = Linear_block(65536, 10)
                    else:
                        input_CNN = Input_CNN_block(3, 64)
                        model = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
                        linear = Linear_block(65536, 10)
                        #print("model structure : ", model_clone[-1])
                    model_results_fitness.append([train_model(model, linear, input_CNN, learning_rate, nsamples,
                                                              load_train_dataset())])  # save training loss in [0]
                    torch.save(input_CNN,
                               "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\input_CNN_s%d_module_%d.pth"%(tracker+1, (population + int((population*0.25))*v + 5*v + time_checker)))
                    torch.save(linear,
                               "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\linear_s%d_module_%d.pth"%(tracker+1, (population + int((population*0.25))*v + 5*v + time_checker)))
                    torch.save(model,
                               "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\model_s%d_module_%d.pth"%(tracker+1, (population + int((population*0.25))*v + 5*v + time_checker)))

                    model_results_fitness[len(model_clone) - 1].append(
                        test_model(model, linear, input_CNN, load_test_dataset(), test_dataset(),
                                   nsamples))  # save test accuracy in [1]


            for i in range(0, len(model_clone)):  # ordering the clone by fitness(accuracy)
                for j in range(0, len(model_clone) - 1 - i):
                    if model_results_fitness[j][1] > model_results_fitness[j + 1][1]:
                        temp_oder = model_results_fitness[j]
                        model_results_fitness[j] = model_results_fitness[j + 1]
                        model_results_fitness[j + 1] = temp_oder

                        temp_oder = model_clone[j]
                        model_clone[j] = model_clone[j + 1]
                        model_clone[j + 1] = temp_oder

                        temp_oder = model_index[j]
                        model_index[j] = model_index[j + 1]
                        model_index[j + 1] = temp_oder

            print("modle index before delate::::", model_index)

            print("In ", v + 1, "th genetation, delate the lowset clone")

            for i in range(0, len(model_clone) - population + 5):  # delate lowest fitness clones based on how many added in the cone pool
                temp_index = 0
                model_results_fitness.pop(0)
                model_clone.pop(0)
                delete_file(int(model_index[0]),tracker+1)######################################################################
                model_index.pop(0)#######################################################################################
            print("clone number after delate: ", len(model_clone))
            print("In ", v + 1, "th genetation, add random clone to keep the pool number")

            print("model index after delete::::::", model_index)

            time_checker = 0
            while time_checker < 5:  # adding 5 new models and fitness(accuracy)
                y = []
                temp_model = [[0] * 2 for row in range(0,5)]
                checksructure = 0
                for j in range(0, NUM_NODES[m]):
                    y.append(random.choice([1, 2, 3, 4, 5, 6, 7, 8]))
                for q in range(0, NUM_NODES[m]):
                    if y[q] == 5 or y[q] == 6 or y[q] == 7 or y[q] == 8 :
                        checksructure = checksructure + 1
                if checksructure == NUM_NODES[m]:
                    y = []
                else:
                    temp_model[time_checker][0] = y
                    y = []
                    for k in range(0, int(NUM_NODES[m] * (NUM_NODES[m] + 1) / 2)):
                        y.append(random.choice([0, 1]))
                    temp_model[time_checker][1] = y
                    model_clone.append(temp_model[time_checker])
                    time_checker = time_checker + 1
                    model_index.append(int(population + int(population * 0.25)*(v+1) + 5*v+ time_checker))  ############################

                    if tracker > 0:
                        current_model = []
                        input_CNN = Input_CNN_block(3, 64)
                        model2 = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
                        higest_model = torch.load("C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")
                        model = connection(higest_model, model2)
                        linear = Linear_block(65536, 10)
                    else:
                        input_CNN = Input_CNN_block(3, 64)
                        model = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
                        print("model structure : ", model_clone[-1])
                        linear = Linear_block(65536, 10)
                    model_results_fitness.append([train_model(model, linear, input_CNN, learning_rate, nsamples,
                                                              load_train_dataset())])  # save training loss in [0]
                    torch.save(input_CNN,
                               "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\input_CNN_s%d_module_%d.pth" % (
                               tracker + 1, int(population + int(population * 0.25)*(v+1) + 5*v+ time_checker)))
                    torch.save(linear,
                               "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\linear_s%d_module_%d.pth" % (
                               tracker + 1, int(population + int(population * 0.25)*(v+1) + 5*v+ time_checker)))
                    torch.save(model,
                               "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\model_s%d_module_%d.pth" % (
                               tracker + 1, int(population + int(population * 0.25)*(v+1) + 5*v+ time_checker)))
                    model_results_fitness[len(model_clone) - 1].append(
                        test_model(model, linear, input_CNN, load_test_dataset(), test_dataset(),
                                   nsamples))  # save test accuracy in [1]


            print("model index after add 5 new :::::", model_index)
            print("the number of models: ",len(model_index))


        for i in range(0, len(model_clone)):  # ordering the clone by fitness(accuracy)
            for j in range(0, len(model_clone) - 1 - i):
                if model_results_fitness[j][1] > model_results_fitness[j + 1][1]:
                    temp_oder = model_results_fitness[j]
                    model_results_fitness[j] = model_results_fitness[j + 1]
                    model_results_fitness[j + 1] = temp_oder

                    temp_oder = model_clone[j]
                    model_clone[j] = model_clone[j + 1]
                    model_clone[j + 1] = temp_oder

                    temp_oder = model_index[j]
                    model_index[j] = model_index[j + 1]
                    model_index[j + 1] = temp_oder

        initial_model = copy.deepcopy(model_clone)
        print("In ", v + 1, "th genetation, save the highest profermence model")
        if tracker > 0:
            higest_result = float(model_results_fitness[-1][1])
            check_model_result.append(float(model_results_fitness[-1][1]))
            highest_model_structure = copy.deepcopy(model_clone[-1])
            higest_symble = model_index[-1]  # get the higheest result index
            model_symbel = copy.deepcopy(model_clone[-1])
            check_model.append(higest_symble)
            # model2 = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
            # higest_model.load_state_dict(
            #     torch.load("C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth"))
            #
            # connected_model = connection(higest_model, model2)
            #check_model.append(connection(higest_model, model2))
            # for i in range(0, len(model_symbel[0])):
            #     highest_model_structure[0].append(model_symbel[0][i])
            # for i in range(0, len(model_symbel[1])):
            #     highest_model_structure[1].append(model_symbel[1][i])
            saved_structure = copy.deepcopy(highest_model_structure)
            check_model_structure.append(highest_model_structure)

            remainDirsList.append("model_s%d_module_%d.pth" % ((tracker + 1), higest_symble))
            delete_all(remainDirsList)

            # torch.save(connected_model.state_dict(),
            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")
            # torch.save(check_model[tracker].state_dict(),
            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\%d_module.pth" % (
            #                int(tracker + 1)))
            # torch.save(check_model[tracker],
            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\check_%d_whole_module.pth" % (
            #                int(tracker + 1)))
            print("highest model index : ", model_index[-1], " and stage is : ", (tracker + 1))
            print("highest symble", higest_symble)
            print("highest result", higest_result)
            print("saved structure", saved_structure)
            print("highest model structure: ", highest_model_structure)
            print("highest model index in all stage:", check_model)
            if tracker > 1:
                if check_model_result[tracker] < check_model_result[tracker - 1]:
                    if check_model_result[tracker] < check_model_result[tracker - 2]:
                        if check_model_result[tracker - 1] > check_model_result[tracker - 2]:
                            final_accuracy = check_model_result[tracker - 1]
                            final_structcure = check_model_structure[tracker - 1]
                            final_model_index = check_model[tracker - 1]
                            # torch.save(final_model.state_dict(),
                            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_module.pth")
                            # torch.save(final_model,
                            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_best_whole_module.pth")
                            print("final model index : ", final_model_index, " and stage is : ", (tracker))
                            print("final accuracy is :", final_accuracy)
                            print("final structure is :", final_structcure)
                            print("highest model index in all stage:", check_model)
                            os._exit()
                        else:
                            final_accuracy = check_model_result[tracker - 2]
                            final_structcure = check_model_structure[tracker - 2]
                            final_model_index = check_model[tracker - 2]
                            # torch.save(final_model.state_dict(),
                            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_module.pth")
                            # torch.save(final_model,
                            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_best_whole_module.pth")
                            print("final model index : ", final_model_index, " and stage is : ", (tracker - 1))
                            print("final accuracy is :", final_accuracy)
                            print("final structure is :", final_structcure)
                            print("highest model index in all stage:", check_model)
                            os._exit()
                    else:
                        pass
                else:
                    final_accuracy = check_model_result[tracker]
                    final_structcure = check_model_structure[tracker]
                    final_model_index = check_model[tracker]
                    if tracker == (len(NUM_NODES) - 1):
                        # torch.save(final_model.state_dict(),
                        #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_%d_module.pth" % (
                        #                int(tracker + 1)))
                        # torch.save(final_model,
                        #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_%d_whole_module.pth"% (
                        #                int(tracker + 1)))
                        print("final model index : ", final_model_index, " and stage is : ", (tracker + 1))
                        print("final accuracy is :", final_accuracy)
                        print("final structure is :", final_structcure)
                        print("highest model index in all stage:", check_model)
                    else:
                        pass


        else:
            higest_result = float(model_results_fitness[-1][1])  # get heighest result in this stage
            check_model_result.append(float(model_results_fitness[-1][1]))  # copy this heighest result
            higest_symble = model_index[-1]  # get the higheest result index
            highest_model_structure = copy.deepcopy(model_clone[-1])  # save the eighet structure
            check_model_structure.append(model_clone[-1])  # copy the highest structure
            saved_structure = copy.deepcopy(highest_model_structure)  # save the structure in the operating arry
            #higest_model = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64,
            #                               64)  # save the model
            check_model.append(higest_symble)
            # torch.save(higest_model.state_dict(),
            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")  # save the weight
            # torch.save(check_model[tracker].state_dict(),
            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\%d_module.pth" % (
            #                int(tracker + 1)))  # copy the weight
            # torch.save(check_model[tracker],
            #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\chech_whole_module.pth")
            remainDirsList.append("model_s%d_module_%d.pth"%((tracker+1), higest_symble))
            delete_all(remainDirsList)

            print("highest model index : ", higest_symble, " and stage is : ", (tracker+1))
            print("highest result", higest_result)
            print("highest model structure: ", highest_model_structure)
            print("saved structure", saved_structure)
            print("highest model index in all stage:", check_model)
        tracker = tracker + 1


# else:
#
#     for m in range(0, len(NUM_NODES)):
#         initial_model = [[0] * 2 for row in range(population)]  # initialize the models
#         model_results_fitness = [[0] * 2 for row in range(population)]
#         model_clone = []
#         model_index = []
#         x = []
#         for i in range(0, population):
#             model_index.append(i+1)
#
#         print("stage", (m + 1), " have ", NUM_NODES[m], "layers")
#         print("start to initialzation the population: ")
#
#         time_checker = 0
#
#         while time_checker < population:
#             checksructure = 0
#             for j in range(0, NUM_NODES[m]):
#                 x.append(random.choice([1, 2, 3, 4, 5, 6, 7, 8]))
#             for q in range(0, NUM_NODES[m]):
#                 if x[q] == 5 or x[q] == 6 or x[q] == 7 or x[q] == 8:
#                     checksructure = checksructure + 1
#             if checksructure == NUM_NODES[m]:
#                 x = []
#             else:
#                 initial_model[time_checker][0] = x
#                 x = []
#                 for k in range(0, int(NUM_NODES[m] * (NUM_NODES[m] + 1) / 2)):
#                     x.append(random.choice([0, 1]))
#                 initial_model[time_checker][1] = x
#                 x = []
#                 time_checker = time_checker + 1
#
#
#         if tracker > 0:
#             current_model = []
#             lenth = 0
#             # for c in range(0, tracker):
#             #     lenth = (1 + NUM_NODES[c]) * NUM_NODES[c] / 2
#             #     current_model.append(CNN_final_block(saved_structure[0][0:NUM_NODES[c]], NUM_NODES[c],
#             #                                          saved_structure[1][0:int(lenth)], 64, 64))
#             #     saved_structure[0] = saved_structure[0][NUM_NODES[c]:]
#             #     saved_structure[1] = saved_structure[1][int(lenth):]
#             #
#             # higest_model = current_model[0]
#             # current_model = current_model[1:]
#             # for c in range(0, tracker - 1):
#             #     higest_model = connection(higest_model, current_model[c])
#             # print("highest model", higest_model)
#             higest_model = torch.load("C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")
#             for i in range(0, population):
#                 input_CNN = Input_CNN_block(3, 64)
#                 model2 = CNN_final_block(initial_model[i][0], NUM_NODES[m], initial_model[i][1], 64, 64)
#                 model = connection(higest_model, model2)
#                 linear = Linear_block(65536, 10)
#                 model_results_fitness[i][0] = train_model(model, linear, input_CNN, learning_rate, nsamples,
#                                                           load_train_dataset())  # save training loss in [0]
#                 torch.save(input_CNN,
#                            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\input_CNN_s%d_module_%d.pth" % (
#                            tracker + 1, i+1))
#                 torch.save(linear,
#                            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\linear_s%d_module_%d.pth" % (
#                            tracker + 1, i+1))
#                 torch.save(model,
#                            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\model_s%d_module_%d.pth" % (
#                                tracker + 1, i + 1))
#
#                 model_results_fitness[i][1] = test_model(model, linear, input_CNN, load_test_dataset(), test_dataset(),
#                                                          nsamples)  # save test accuracy in [1]
#         else:
#             for i in range(0, population):
#                 input_CNN = Input_CNN_block(3, 64)
#                 model = CNN_final_block(initial_model[i][0], NUM_NODES[m], initial_model[i][1], 64, 64)
#                 linear = Linear_block(65536, 10)
#                 print("model structure : ", initial_model[i])
#                 model_results_fitness[i][0] = train_model(model, linear, input_CNN, learning_rate, nsamples,
#                                                           load_train_dataset())  # save training loss in [0]
#                 torch.save(input_CNN,
#                            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\input_CNN_s%d_module_%d.pth"%(tracker+1, i+1))
#                 torch.save(linear,
#                            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\linear_s%d_module_%d.pth"%(tracker+1, i+1))
#                 torch.save(model,
#                            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\model_s%d_module_%d.pth" % (
#                                tracker + 1, i + 1))
#                 model_results_fitness[i][1] = test_model(model, linear, input_CNN, load_test_dataset(), test_dataset(),
#                                                          nsamples)  # save test accuracy in [1]
#
#         print("clone the population to clone pool")
#         model_clone = copy.deepcopy(initial_model)  # clone initial models for immune operations
#         print("start to keep upload the generations")
#
#         for v in range(0, generation):
#             highest_clone = []
#             highest_clone_fitness = []
#             highest_clone_index = []
#             delete = []
#             print("this is the ", v + 1, "th generation")
#
#             print("delete the same structrue models")
#             for r1 in range(0, len(model_clone) - 1):
#                 for r2 in range(r1 + 1, len(model_clone)):
#                     checker = 0
#                     point = 0
#                     for r3 in range(0, len(model_clone[0][0])):
#                         if model_clone[r1][1][-(r3 + 1)] == model_clone[r2][1][-(r3 + 1)]:
#                             if model_clone[r1][1][-(r3 + 1)] == 1:
#                                 checker = checker + 1
#                                 if model_clone[r1][0][-(r3 + 1)] == model_clone[r2][0][-(r3 + 1)]:
#                                     point = point + 1
#                                 else:
#                                     pass
#                             else:
#                                 pass
#                         else:
#                             pass
#                     num_one = 0
#                     for r4 in range(0, len(model_clone[r1][0])):
#                         if model_clone[r1][0][r4] == 1:
#                             num_one = num_one + 1
#                     if checker == num_one and point == num_one:
#                         if model_results_fitness[r1] > model_results_fitness[r2]:
#                             delete.append(r2)
#                         elif model_results_fitness[r1] < model_results_fitness[r2]:
#                             delete.append(r1)
#                     else:
#                         pass
#             delete = list(set(delete))
#             delete.sort()
#             if len(delete) >= 5:
#                 for r5 in range(0, 5):
#                     model_clone.pop((delete[r5] - r5))
#                     model_results_fitness.pop((delete[r5] - r5))
#             else:
#                 for r6 in range(0, len(delete)):
#                     model_clone.pop((delete[r6] - r6))
#                     model_results_fitness.pop((delete[r6] - r6))
#                 for r7 in range(0, 5 - len(delete)):
#                     model_clone.pop(r7)
#                     model_results_fitness.pop(r7)
#
#
#             for i in range(0, len(model_clone)):  # ordering the clone by fitness(accuracy) to
#                 for j in range(0, len(model_clone) - 1 - i):  # select the highest 25% in to
#                     if model_results_fitness[j][1] > model_results_fitness[j + 1][1]:
#                         temp_oder = model_results_fitness[j]
#                         model_results_fitness[j] = model_results_fitness[j + 1]
#                         model_results_fitness[j + 1] = temp_oder
#
#                         temp_oder = model_clone[j]
#                         model_clone[j] = model_clone[j + 1]
#                         model_clone[j + 1] = temp_oder
#
#                         temp_oder = model_index[j]
#                         model_index[j] = model_index[j + 1]
#                         model_index[j +1] = temp_oder
#             print("In ", v + 1, "th genetation, get the highest clones")
#             for i in range(0, int(len(model_clone) * 0.25)+1):  # select highest 25% clone and save them
#                 highest_clone.append(model_clone[-(i + 1)])
#                 highest_clone_fitness.append(model_results_fitness[-(i + 1)])
#                 highest_clone_index.append(model_index[-(i + 1)])
#             print("highest clones:", (int(len(model_clone) * 0.25)), "clone numbers: ", len(model_clone))
#             print("In ", v + 1, "th genetation, mutate highest clones")
#
#
#             time_checker = 0
#             # temp_model = [[0] * 2 for row in range(len(highest_clone))]
#             while time_checker < len(highest_clone):  # mutate the highest clone pool by affinity
#                 y = []  # calculate affnity and add in to the arry
#                 checksructure = 0
#
#                 y.append(structure_mutate(highest_clone,highest_clone_fitness,len(highest_clone),population,time_checker))##########更改为mutate_structure
#
#                 for q in range(0, NUM_NODES[m]):
#                     if y[0][q] == 5 or y[0][q] == 6 or y[0][q] == 7 or y[0][q] == 8 :
#                         checksructure = checksructure + 1
#                 if checksructure == NUM_NODES[m]:
#                     y = []
#                 else:
#                     # temp_model[time_checker] = y
#                     # y = []
#                     # for k in range(0, int(NUM_NODES[m] * (NUM_NODES[m] + 1) / 2)):
#                     #     y.append(random.choice([0, 1]))
#                     # temp_model[time_checker][1] = y
#                     model_clone.append(y)
#                     time_checker  = time_checker + 1
#                     model_index.append((population + int((population*0.25)*v) + 5*v + time_checker))############################
#
#                     if tracker > 0:
#                         current_model = []
#                         input_CNN = Input_CNN_block(3, 64)
#                         model2 = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
#                         higest_model = torch.load("C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")
#                         model = connection(higest_model, model2)
#                         linear = Linear_block(65536, 10)
#                     else:
#                         input_CNN = Input_CNN_block(3, 64)
#                         model = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
#                         linear = Linear_block(65536, 10)
#                         print("model structure : ", model_clone[-1])
#                     model_results_fitness.append([train_model(model, linear, input_CNN, learning_rate, nsamples,
#                                                               load_train_dataset())])  # save training loss in [0]
#                     torch.save(input_CNN,
#                                "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\input_CNN_s%d_module_%d.pth"%(tracker+1, (population + int((population*0.25)*v) + 5*v + time_checker)))
#                     torch.save(linear,
#                                "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\linear_s%d_module_%d.pth"%(tracker+1, (population + int((population*0.25)*v) + 5*v + time_checker)))
#                     torch.save(model,
#                                "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\model_s%d_module_%d.pth"%(tracker+1, (population + int((population*0.25)*v) + 5*v + time_checker)))
#
#                     model_results_fitness[len(model_clone) - 1].append(
#                         test_model(model, linear, input_CNN, load_test_dataset(), test_dataset(),
#                                    nsamples))  # save test accuracy in [1]
#
#
#             for i in range(0, len(model_clone)):  # ordering the clone by fitness(accuracy)
#                 for j in range(0, len(model_clone) - 1 - i):
#                     if model_results_fitness[j][1] > model_results_fitness[j + 1][1]:
#                         temp_oder = model_results_fitness[j]
#                         model_results_fitness[j] = model_results_fitness[j + 1]
#                         model_results_fitness[j + 1] = temp_oder
#
#                         temp_oder = model_clone[j]
#                         model_clone[j] = model_clone[j + 1]
#                         model_clone[j + 1] = temp_oder
#
#                         temp_oder = model_index[j]
#                         model_index[j] = model_index[j + 1]
#                         model_index[j + 1] = temp_oder
#
#             print("modle index before delate::::", model_index)
#
#             print("In ", v + 1, "th genetation, delate the lowset clone")
#             for i in range(0, len(model_clone) - population + 5):  # delate lowest fitness clones based on how many added in the cone pool
#                 model_results_fitness.pop(0)
#                 model_clone.pop(0)
#                 delete_file(model_index[i],tracker+1)######################################################################
#                 model_index.pop(0)#######################################################################################
#             print("clone number after delate: ", len(model_clone))
#             print("In ", v + 1, "th genetation, add random clone to keep the pool number")
#
#             print("model index after delete::::::", model_index)
#
#             time_checker = 0
#             while time_checker < 5:  # adding 5 new models and fitness(accuracy)
#                 y = []
#                 temp_model = [[0] * 2 for row in range(0,5)]
#                 checksructure = 0
#                 for j in range(0, NUM_NODES[m]):
#                     y.append(random.choice([1, 2, 3, 4, 5, 6, 7, 8]))
#                 for q in range(0, NUM_NODES[m]):
#                     if y[q] == 5 or y[q] == 6 or y[q] == 7 or y[q] == 8 :
#                         checksructure = checksructure + 1
#                 if checksructure == NUM_NODES[m]:
#                     y = []
#                 else:
#                     temp_model[time_checker][0] = y
#                     y = []
#                     for k in range(0, int(NUM_NODES[m] * (NUM_NODES[m] + 1) / 2)):
#                         y.append(random.choice([0, 1]))
#                     temp_model[time_checker][1] = y
#                     model_clone.append(temp_model[time_checker])
#                     time_checker = time_checker + 1
#                     model_index.append(int(population + (population * 0.25)*(v+1) + 5*v + time_checker))  ############################
#
#                     if tracker > 0:
#                         current_model = []
#                         input_CNN = Input_CNN_block(3, 64)
#                         model2 = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
#                         higest_model = torch.load("C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")
#                         model = connection(higest_model, model2)
#                         linear = Linear_block(65536, 10)
#                     else:
#                         input_CNN = Input_CNN_block(3, 64)
#                         model = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
#                         print("model structure : ", model_clone[-1])
#                         linear = Linear_block(65536, 10)
#                     model_results_fitness.append([train_model(model, linear, input_CNN, learning_rate, nsamples,
#                                                               load_train_dataset())])  # save training loss in [0]
#                     torch.save(input_CNN,
#                                "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\input_CNN_s%d_module_%d.pth" % (
#                                tracker + 1, int(population + (population * 0.25)*(v+1) + 5*v + time_checker)))
#                     torch.save(linear,
#                                "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\rest_module\\linear_s%d_module_%d.pth" % (
#                                tracker + 1, int(population + (population * 0.25)*(v+1) + 5*v + time_checker)))
#                     torch.save(model,
#                                "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\model_s%d_module_%d.pth" % (
#                                tracker + 1, int(population + (population * 0.25)*(v+1) + 5*v + time_checker)))
#                     model_results_fitness[len(model_clone) - 1].append(
#                         test_model(model, linear, input_CNN, load_test_dataset(), test_dataset(),
#                                    nsamples))  # save test accuracy in [1]
#
#
#             print("model index after add 5 new :::::", model_index)
#
#
#         for i in range(0, len(model_clone)):  # ordering the clone by fitness(accuracy)
#             for j in range(0, len(model_clone) - 1 - i):
#                 if model_results_fitness[j][1] > model_results_fitness[j + 1][1]:
#                     temp_oder = model_results_fitness[j]
#                     model_results_fitness[j] = model_results_fitness[j + 1]
#                     model_results_fitness[j + 1] = temp_oder
#
#                     temp_oder = model_clone[j]
#                     model_clone[j] = model_clone[j + 1]
#                     model_clone[j + 1] = temp_oder
#
#                     temp_oder = model_index[j]
#                     model_index[j] = model_index[j + 1]
#                     model_index[j + 1] = temp_oder
#
#         initial_model = copy.deepcopy(model_clone)
#         print("In ", v + 1, "th genetation, save the highest profermence model")
#         if tracker > 0:
#             higest_result = float(model_results_fitness[-1][1])
#             check_model_result.append(float(model_results_fitness[-1][1]))
#             highest_model_structure = copy.deepcopy(model_clone[-1])
#             higest_symble = model_index[-1]  # get the higheest result index
#             model_symbel = copy.deepcopy(model_clone[-1])
#             check_model.append(higest_symble)
#             # model2 = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64, 64)
#             # higest_model.load_state_dict(
#             #     torch.load("C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth"))
#             #
#             # connected_model = connection(higest_model, model2)
#             #check_model.append(connection(higest_model, model2))
#             # for i in range(0, len(model_symbel[0])):
#             #     highest_model_structure[0].append(model_symbel[0][i])
#             # for i in range(0, len(model_symbel[1])):
#             #     highest_model_structure[1].append(model_symbel[1][i])
#             saved_structure = copy.deepcopy(highest_model_structure)
#             check_model_structure.append(highest_model_structure)
#
#             remainDirsList.append("model_s%d_module_%d.pth" % ((tracker + 1), higest_symble))
#             delete_all(remainDirsList)
#
#             # torch.save(connected_model.state_dict(),
#             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")
#             # torch.save(check_model[tracker].state_dict(),
#             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\%d_module.pth" % (
#             #                int(tracker + 1)))
#             # torch.save(check_model[tracker],
#             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\check_%d_whole_module.pth" % (
#             #                int(tracker + 1)))
#             print("highest model index : ", model_index[-1], " and stage is : ", (tracker + 1))
#             print("highest symble", higest_symble)
#             print("highest result", higest_result)
#             print("saved structure", saved_structure)
#             print("highest model structure: ", highest_model_structure)
#             print("highest model index in all stage:", check_model)
#             if tracker > 1:
#                 if check_model_result[tracker] < check_model_result[tracker - 1]:
#                     if check_model_result[tracker] < check_model_result[tracker - 2]:
#                         if check_model_result[tracker - 1] > check_model_result[tracker - 2]:
#                             final_accuracy = check_model_result[tracker - 1]
#                             final_structcure = check_model_structure[tracker - 1]
#                             final_model_index = check_model[tracker - 1]
#                             # torch.save(final_model.state_dict(),
#                             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_module.pth")
#                             # torch.save(final_model,
#                             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_best_whole_module.pth")
#                             print("final model index : ", final_model_index, " and stage is : ", (tracker))
#                             print("final accuracy is :", final_accuracy)
#                             print("final structure is :", final_structcure)
#                             print("highest model index in all stage:", check_model)
#                             os._exit()
#                         else:
#                             final_accuracy = check_model_result[tracker - 2]
#                             final_structcure = check_model_structure[tracker - 2]
#                             final_model_index = check_model[tracker - 2]
#                             # torch.save(final_model.state_dict(),
#                             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_module.pth")
#                             # torch.save(final_model,
#                             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_best_whole_module.pth")
#                             print("final model index : ", final_model_index, " and stage is : ", (tracker - 1))
#                             print("final accuracy is :", final_accuracy)
#                             print("final structure is :", final_structcure)
#                             print("highest model index in all stage:", check_model)
#                             os._exit()
#                     else:
#                         pass
#                 else:
#                     final_accuracy = check_model_result[tracker]
#                     final_structcure = check_model_structure[tracker]
#                     final_model_index = check_model[tracker]
#                     if tracker == (len(NUM_NODES) - 1):
#                         # torch.save(final_model.state_dict(),
#                         #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_%d_module.pth" % (
#                         #                int(tracker + 1)))
#                         # torch.save(final_model,
#                         #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\final_%d_whole_module.pth"% (
#                         #                int(tracker + 1)))
#                         print("final model index : ", final_model_index, " and stage is : ", (tracker + 1))
#                         print("final accuracy is :", final_accuracy)
#                         print("final structure is :", final_structcure)
#                         print("highest model index in all stage:", check_model)
#                     else:
#                         pass
#
#
#         else:
#             higest_result = float(model_results_fitness[-1][1])  # get heighest result in this stage
#             check_model_result.append(float(model_results_fitness[-1][1]))  # copy this heighest result
#             higest_symble = model_index[-1]  # get the higheest result index
#             highest_model_structure = copy.deepcopy(model_clone[-1])  # save the eighet structure
#             check_model_structure.append(model_clone[-1])  # copy the highest structure
#             saved_structure = copy.deepcopy(highest_model_structure)  # save the structure in the operating arry
#             #higest_model = CNN_final_block(model_clone[-1][0], NUM_NODES[m], model_clone[-1][1], 64,
#             #                               64)  # save the model
#             check_model.append(higest_symble)
#             # torch.save(higest_model.state_dict(),
#             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\highest_module.pth")  # save the weight
#             # torch.save(check_model[tracker].state_dict(),
#             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\%d_module.pth" % (
#             #                int(tracker + 1)))  # copy the weight
#             # torch.save(check_model[tracker],
#             #            "C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module\\chech_whole_module.pth")
#             remainDirsList.append("model_s%d_module_%d.pth"%((tracker+1), higest_symble))
#             delete_all(remainDirsList)
#
#             print("highest model index : ", higest_symble, " and stage is : ", (tracker+1))
#             print("highest result", higest_result)
#             print("highest model structure: ", highest_model_structure)
#             print("saved structure", saved_structure)
#             print("highest model index in all stage:", check_model)
#         tracker = tracker + 1
#
#
#
#
#
#
#
#
