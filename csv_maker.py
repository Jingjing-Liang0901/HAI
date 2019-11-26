from __future__ import print_function, division
import torch
import torch.nn.functional as F
import cv2 # computer vision library (counts no of frames, height, weight)
import os
import numpy as np
#import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import torchaudio
#from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd
import skvideo.io
import skvideo.datasets

def shapemaker(name):
  vdata = skvideo.io.vread(name, as_grey = True)
  varray = np.array(vdata).astype(float)
  vtensor = torch.tensor(varray)

  # T = number of frames
  # M = height
  # N = width
  # C = depth (grayscale)

  diff = 114 - int(vtensor.shape[0])
  diff_array = np.zeros((diff, 224, 224, 1)).astype(float)
  diff_tensor = torch.tensor(diff_array)

  final = torch.cat([vtensor, diff_tensor])

  return final

mp4List = [file for file in glob.glob('**/*.mp4', recursive=True)]
wavList = [file.replace('.mp4', '.wav') for file in mp4List]
txtList = [file.replace('.mp4', '.txt') for file in mp4List]


allVideos = [shapemaker(i).numpy() for i in mp4List[0:3]]

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

allAudios = []
for i in range(len([1, 2, 3])):
  file = wavList[i]
  print(file)
  waveform, sample_rate = torchaudio.load(file)  # load tensor from file
  #print(waveform.shape)
# Resampling to 16khz
  new_sample_rate = 16000

# Padding figure = 73920

# Since Resample applies to a single channel, we resample first channel here
  channel = 0
  transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)

# resampled shape
  #print(transformed.shape)
  #plt.figure()
  #plt.plot(transformed[0,:])
  audioPadded = torch.cat([transformed, torch.zeros(2,73920-transformed.shape[1])],dim=1)
  specgram = torchaudio.transforms.Spectrogram()(audioPadded) # Spectrogram() is a transformation---log spectrogram.
  allAudios.append(specgram.numpy())
  #plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')
  #print(type(specgram))







df = pd.DataFrame(data={"Audio": allAudios,"Video": allVideos})
df.to_csv("avInput.csv", sep=',',index=False)
