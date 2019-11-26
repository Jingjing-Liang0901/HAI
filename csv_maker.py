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
    torch.save(final, 'video.pt')
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
    torch.save(specgram, 'audio.pt')


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

for i in range(0,3):
  getVideo(i)
  getAudio(i)
  getLabel(i)

'''
To check
'''
test = torch.load('video.pt')
print(test.shape)
