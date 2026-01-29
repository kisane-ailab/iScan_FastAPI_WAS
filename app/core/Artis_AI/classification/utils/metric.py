

import torch
import os
import numpy as np

import torch.nn.functional as F

from PIL import Image

def validation_accuracy(model, loader, device, wrong_image=False):
    total = 0
    correct = 0
    a = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (img0, img1, img2, targets) in enumerate(loader):
            img0, img1, img2, targets = img0.to(device), img1.to(device), img2.to(device), targets.to(device)
            outputs = model(img0, img1, img2)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            # correct += predicted.eq(targets).sum().item()
            
            for i, p in enumerate(predicted):
                # if not p == targets[i]:
                idx = batch_idx * 16 + i
                imgnames = loader.dataset.imgpath[idx]
                bboxes = loader.dataset.imgbbox[idx]
                dictionary = loader.dataset.target_dictionary
                for j, imgname in enumerate(imgnames):
                    b = 0
                    img = Image.open(imgname)
                    # w =  list(dictionary.keys())[list(dictionary.values()).index(p)]
                    if p <=130:
                        w = p+1
                    if p>130:
                        w = p-130+1000
                    t = list(dictionary.keys())[list(dictionary.values()).index(targets[i])]
                    if w == t:
                        correct +=1
                    elif w != t:
                        if wrong_image:
                            os.makedirs('./wrong_image/{}'.format(a), exist_ok=True) 
                            img.save('./wrong_image/{}/{}_gt_{}_pred_{}_{}'.format(a, j, t, w, imgname.split('/')[-1]))
                            
                            crop_img = img.crop(bboxes[j])
                            crop_img.save('./wrong_image/{}/{}_cropgt_{}_pred_{}_{}'.format(a, j, t, w, imgname.split('/')[-1]))
                        # print(loader.dataset.imgpath[i])
                if w != t:
                    a += 1

    correct = correct/3
    print(correct, total-correct, total)
    valid_accuracy = correct/total
    return valid_accuracy

def calculate_prob(model, loader, device, returnTarget = True):
    model.eval()

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs) * 76.18
            output = torch.softmax(output, dim=1)
            prob, _ = output.max(1) 

            outputs.append(prob)
            targets.append(target)
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    if returnTarget:
        return outputs, targets
    else:
        return outputs

def ece_score(py, y_test, n_bins=100):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)