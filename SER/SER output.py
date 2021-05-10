import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


import pyaudio
import wave

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

filename = 'model_ser.sav'
loaded_model = pickle.load(open(filename, 'rb'))



RECORD_SECONDS = 3

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,channels=1,rate=44100, input=True, frames_per_buffer=1024)

print("**** recording")

frames = []

for i in range(0, int(44100 / 1024 * RECORD_SECONDS)):
    data = stream.read(1024)
    frames.append(data)

print("**** done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open('output.wav', 'wb')
wf.setnchannels(2)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(44100)
wf.writeframes(b''.join(frames))
wf.close()

data, sample_rate = librosa.load('output.wav')
data = np.array(data)

plt.figure(figsize=(15, 3))
librosa.display.waveplot(data, sr=sample_rate)

X=data
stft=np.abs(librosa.stft(X))
result=np.array([])
mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
result=np.hstack((result, mfccs))
chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
result=np.hstack((result, chroma))
mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
result=np.hstack((result, mel))
feature = result
feature

input = []
input.append(feature)

input= np.array(input)

output = loaded_model.predict(input)
print(input)

while True:
    print(output)
