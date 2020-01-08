#!bin/env python
import os
import os.path
import shutil

"delete function to remove the target models"
def delete_file(index,tracker):

    input_file = r"C:\Users\garychenai\Desktop\Final project\codes\module\rest_module\input_CNN_s%d_module_%d.pth"%(tracker,index)
    linear_file = r"C:\Users\garychenai\Desktop\Final project\codes\module\rest_module\linear_s%d_module_%d.pth"%(tracker, index)
    model_file = r"C:\Users\garychenai\Desktop\Final project\codes\module\model_s%d_module_%d.pth"%(tracker, index)
    os.remove(input_file)
    os.remove(linear_file)
    os.remove(model_file)
    print("Successfully delete input_CNN_s%d_module_%d.pth and linear_s%d_module_%d.pth and model_s%d_module_%d.pth"%(tracker, index,tracker, index, tracker, index))

"delete function to remove the useless models by name"
def delete_all(remainDirsList):
    path = 'C:\\Users\\garychenai\\Desktop\\Final project\\codes\\module'

    dirsList = []
    dirsList = os.listdir(path)
    for i in dirsList:
        if i == "rest_module":
            dirsList.remove("rest_module")
    for f in dirsList:
        if f not in remainDirsList:
            filepath = os.path.join(path, f)
            os.remove(filepath)

    tempname = os.path.join(path, 'copy.pth')
    tempList = os.listdir(path)
    shutil.copy(os.path.join(path, remainDirsList[-1]), tempname)
    newList = os.listdir(path)
    for f in newList:
        if f == 'copy.pth':
            os.rename(os.path.join(path, f), os.path.join(path, "highest_module.pth"))



