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


#allVideos = [shapemaker(i) for i in mp4List[0:3]]


'''
for i in range(len([1, 2, 3])):
  file = mp4List[i]
  print(file)
  channels = 3
  cap = cv2.VideoCapture(file)
  print(cap)
  nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print(nFrames)
  print(width)
  print(height)

  # This is not good!
  frames = torch.FloatTensor(channels, nFrames, width, height)
  print(frames)

# concatenate frames and zero matrix.
  framesPadded = torch.cat([frames, torch.zeros(channels,114-nFrames,width,height)],dim=1)
# you probably want to downsample this, otherwise the dataset will be huge
  framesSmall = F.interpolate(framesPadded,size=[64,64])
  #print(framesSmall)
  allVideos.append(framesSmall)
'''

#allAudios = []
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
exporting

import csv

with open('employee_file2.csv', mode='w') as csv_file:
    fieldnames = ['emp_name', 'dept', 'birth_month']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})
    writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})
'''

with open('1125.csv', mode='w') as csv_file:
    fieldnames = ['video', 'audio', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(0,3):
        writer.writerow({'video': getVideo(i), 'audio': getAudio(i), 'label': getLabel(i)})

#df = pd.DataFrame(data={"Audio": allAudios,"Video": allVideos})
#df.to_csv("avInput.csv", sep=',',index=False)
