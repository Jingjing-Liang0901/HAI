from __future__ import print_function, division
import torch
import torch.nn.functional as F
#import cv2 # computer vision library (counts no of frames, height, weight)
import os
import numpy as np
#import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import torchaudio
#from scipy.io.wavfile import read
import scipy.io.wavfile as wav
#import matplotlib.pyplot as plt
#import pandas as pd
import skvideo.io
import skvideo.datasets
import csv

'''
Data prep
'''

mp4List = [file for file in glob.glob('**/*.mp4', recursive=True)]
wavList = [file.replace('.mp4', '.wav') for file in mp4List]
txtList = [file.replace('.mp4', '.txt') for file in mp4List]

def getVideo(index):
    filename = mp4List[index]
    vdata = skvideo.io.vread(name, as_grey = True)
    varray = np.array(vdata).astype(float)
    vtensor = torch.tensor(varray)
    diff = 114 - int(vtensor.shape[0])
    diff_array = np.zeros((diff, 224, 224, 1)).astype(float)
    diff_tensor = torch.tensor(diff_array)
    final = torch.cat([vtensor, diff_tensor])
    return final

'''
 # T = number of frames
 # M = height
 # N = width
 # C = depth (grayscale)
 '''

'''
Padding figure = 73920
'''
def getAudio(index):
    filename = wavList[index]
    waveform, sample_rate = torchaudio.load(filename)
    new_sample_rate = 16000
    channel = 0
    transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)
    audioPadded = torch.cat([transformed, torch.zeros(2,73920-transformed.shape[1])],dim=1)
    specgram = torchaudio.transforms.Spectrogram()(audioPadded)
    return specgram


  #allAudios.append(specgram)
  #plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')
  #print(type(specgram))

def getLabel(index):
    filename = txtList[index]
    label = open(filename, "r").read().split()[1:-2])
    return label


'''
The Loop
'''
allAudios = []
allVideos = []
allLabels = []
for i in range(0,3):
  allVideos.append(getVideo(i))
  allAudios.append(getAudio(i))
  allLabels.append(getLabel(i))

torch.save(final, 'video.pt')
torch.save(specgram, 'audio.pt')

'''
To check
'''
test = torch.load('video.pt')
print(test.shape)



'''
Calculate the error
'''

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])
