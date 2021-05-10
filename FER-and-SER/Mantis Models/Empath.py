import pyttsx3
import datetime
import speech_recognition as sr
import soundfile
import numpy as np
import pandas as pd
import pyaudio
import wave
import librosa
from playsound import playsound
from keras.models import model_from_json
from keras.models import load_model
from cv2 import cv2
from sklearn.preprocessing import OneHotEncoder
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)



engine= pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[3].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishMe():
    speak("Hey there!")
    hour= int(datetime.datetime.now().hour)
    if hour>=4 and hour<=12:
        speak("Good Morning ")
    elif hour>=12 and hour<=16:
        speak("Good After Noon ")
    else:
        speak("Good Evening ")
    speak("This is Mantis")


def takeCommand():
    #take microphone input from user and returns string output
    r=sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        playsound("F:\\MANTIS\\FER and SER\\beep.mp3")
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio)
        print(f"User said: {query}\n")

    except Exception as e:
        #print(e) this will print the error

        speak("Unable to recognise voice")
        return "0"

    return query

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    frames = []
    p = pyaudio.PyAudio()    
    stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
    playsound("F:\\MANTIS\\FER and SER\\beep.mp3")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    speak("voice recorded")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open("F:\\MANTIS\\FER and SER\\output.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    sound_file = 'F:\\MANTIS\\FER and SER\\output.wav'
    return sound_file
    
def extract_mfcc(wav_file_name):
    #This function extracts mfcc features and obtain the mean of each dimension
    #Input : path_to_wav_file
    #Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)    
    return mfccs


def predict_emotions_audio():

    json_file = open('F:\\MANTIS\\FER and SER\\Saved Models\\model_LSTM.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("F:\\MANTIS\\FER and SER\\Saved Models\\model_LSTM.h5")
    loaded_model.save('F:\\MANTIS\\FER and SER\\Saved Models\\model_LSTM.hdf5')
    loaded_model=load_model('F:\\MANTIS\\FER and SER\\Saved Models\\model_LSTm.hdf5', compile=False)

    sound_file = record_audio()

    df = pd.read_csv("F:\\MANTIS\\FER and SER\\mfcc features.csv")
    Y = df['Emotions'].values
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()   
    
    feature=[]
    feature.append(extract_mfcc(sound_file))
    feature_out = pd.DataFrame(feature)
    fet = feature_out.iloc[: ,:].values
    y = loaded_model.predict(np.expand_dims(fet, -1))
    output = str(encoder.inverse_transform(y))
    output1=output[3:-3]
    return output1


def predict_emotions_video():

    json_file = open('F:\\MANTIS\\FER and SER\\Saved Models\\model_num.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("F:\\MANTIS\\FER and SER\\Saved Models\\model_num.h5")
    loaded_model.save('F:\\MANTIS\\FER and SER\\Saved Models\\model_num.hdf5')
    loaded_model=load_model('F:\\MANTIS\\FER and SER\\Saved Models\\model_num.hdf5', compile=False)

    maxindex=-1
    cv2.ocl.setUseOpenCL(False)
    emotion_dict = {-1:"Unable to recognize face",0: "angry", 1: "disgusted", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}
    cap = cv2.VideoCapture(0)
    i=0
    List=[]
    while (i<100):
        ret, frame = cap.read()
        if  not ret:
            break
        bounding_box = cv2.CascadeClassifier('F:\MANTIS\FER and SER\FER\Harcascade\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = loaded_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            List.append(maxindex)
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
        
          

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        i=i+1
    cv2.destroyAllWindows()
    if maxindex == -1:
        output2 = emotion_dict[maxindex]
    else:
        counter = 0
        num = List[0]
        for i in List: 
            curr_frequency = List.count(i) 
            if(curr_frequency> counter): 
                counter = curr_frequency 
                num = i 
        output2 = emotion_dict[num]
    return str(output2)

def sad():
    p=0
    speak("i can help you by giving some advices. Do you want to hear?")
    while(p==0):
        query=takeCommand().lower()
        if 'yes' in query or 'right' in query or 'sure' in query:
            speak("1 you can go for a walk and take some fresh air or do exercise, it will help you in changing your mood")
            speak("2 you can listen to some good music or do any other hobby, it will help in distracting your mind from that mood") 
            speak("3 nuture yourself with good nutrition, you should not take risk with your health")
            speak("4 you can express your feelings to someone, it will make you feel light")
            speak("if you still feel sad you must visit a psychatrist, sometimes problem is big than it seems")
            p=1

        elif 'no' in query or 'not' in query or 'leave' in query:
            speak("take care")
            p=1
        
        else:
            speak("sorry! didn't get you. say that again please, should I give you advice?")
            
    




def angry():
    p=0
    speak("i can help you by giving some advices. Do you want to hear?")
    while(p==0):
        query=takeCommand().lower()
        if 'yes' in query or 'right' in query or 'sure' in query:
            speak("1 meditation is a very good way for reducing anger")
            speak("2 distract yourself by doing some work you like, it can be cooking, writing, drawing or any other activity") 
            speak("3 nuture yourself with good nutrition, you should not take risk with your health")
            speak("4 Go otside and have some fresh air, this will reduce stress on mind")
            speak("if nothing works try visiting a doctor or anger management classes, this can help for sure")
            p=1

        elif 'no' in query or 'not' in query or 'leave' in query:
            speak("take care")
            p=1
        else:
            speak("sorry! didn't get you. say that again please, should I give you advice?")
            



def neutral():
    p=0
    speak("Being calm and neutral is the best way to maintain emotional stability, but sometimes this can make a person miserable or depressing. I can help you giving advice that can cheer up your mood . Do you want to hear?")
    while(p==0):
        query=takeCommand().lower()
        if 'yes' in query or 'right' in query or 'sure' in query:
            speak("1 Listening good music can often cheerup mood")
            speak("2 keeping yourself busy with hobbies and doing things you like") 
            speak("3 Talking and sharing thoughts with someone")
            speak("4 Helping someone can do.")
            p=1

        elif 'no' in query or 'not' in query or 'leave' in query:
            speak("take care")
            p=1
        else:
            speak("sorry! didn't get you. say that again please, should I give you advice?")
            



def happy():
    p=0
    speak("feeling happy and joyful is the best emotion, i can help you by giving some advices that can make you even more cheerful. Do you want to hear?")
    while(p==0):
        query=takeCommand().lower()
        if 'yes' in query or 'right' in query or 'sure' in query:
            speak("1 Share your feelings with others, remember to express your feelings in appropreate ways.")
            speak("2 You can do some work that you enjoy doing") 
            speak("3 do some fun activities with friends like playing or watching movie or anything else") 
            speak("4 Have some fresh air while taking a walk or playing outside")
            speak("Make sure to have a positive attitude, negitivity often let you down")
            p=1

        elif 'no' in query or 'not' in query or 'leave' in query:
            speak("take care")
            p=1
        else:
            speak("sorry! didn't get you. say that again please, should I give you advice?")



def disgust():
    p=0
    speak("disgust mood often let people down, i can help you by giving some advices to deal with it. Do you want to hear?")
    while(p==0):
        query=takeCommand().lower()
        if 'yes' in query or 'right' in query or 'sure' in query:
            speak("1 meditation is a very good for overcoming such mood")
            speak("2 distract yourself by doing some work you like, it can be cooking, writing, drawing or any other activity") 
            speak("3 nuture yourself with good nutrition and maintain proper sanitization")
            speak("4 Go otside and have some fresh air, this will reduce stress on mind")
            speak("if nothing works try visiting a doctor, sometimes this mood can be depressing")
            p=1

        elif 'no' in query or 'not' in query or 'leave' in query:
            speak("take care")
            p=1
        else:
            speak("sorry! didn't get you. say that again please, should I give you advice?")



def fear():
    p=0
    speak("You can learn to feel less fearful and to cope with fear, i can help you by giving some advices. Do you want to hear?")
    while(p==0):
        query=takeCommand().lower()
        if 'yes' in query or 'right' in query or 'sure' in query:
            speak("1 Stay engange in some work that you like")
            speak("2 Eat healthy") 
            speak("3 nTry to relax your mind by doing yoga or meditation")
            speak("4 Go otside and have some fresh air, this will reduce stress on mind")
            speak("This mood can sometimes be serious, in that case you should visit a doctor")
            p=1
        elif 'no' in query or 'not' in query or 'leave' in query:
            speak("take care")
            p=1
        else:
            speak("sorry! didn't get you. say that again please, should I give you advice?")


def surprise():
    p=0
    speak(" Surprise is anytime that you were wrong and your brain tells you about it. Do you want to hear some advice on what should you do when you feel surprised?")
    while(p==0):
        query=takeCommand().lower()
        if 'yes' in query or 'right' in query or 'sure' in query:
            speak("1 When we're surprised, for better or for worse, our emotions intensify up to 400 percent. meditation is a good way to remain emotionally stable")
            speak("2 dIf we’re surprised with something positive, we’ll feel more intense feelings of happiness or joy than we normally would had absent the surprise. Similarly, if we’re surprised by something negative, our feelings of anger, despair or unhappiness will also intensify because of the surprise, In such case make sure to have positive vibes ") 
            p=1

        elif 'no' in query or 'not' in query or 'leave' in query:
            speak("take care")
            p=1
        else:
            speak("sorry! didn't get you. say that again please, should I give you advice?")


if __name__ == "__main__":
    
    wishMe()
    speak("How are you")
    emotion_audio =(predict_emotions_audio())
    emotion_video = (predict_emotions_video())



    speak("predicted emotion from audio is")
    speak(emotion_audio)
    speak("predicted emotion from video is")
    speak(emotion_video)
    emotion_list=[emotion_video,emotion_audio]
    print(emotion_list)
    p=0

    while(p==0):
        if 'sad' in emotion_list:
            speak("are you feeling sad or feeling low")
            query=takeCommand().lower()
        
            if 'yes' in query or 'right' in query:
                sad()
                p=1
                
            elif 'no' in query or 'not' in query or 'wrong' in query:
                speak("please tell me what are you feeling")
                query1=takeCommand().lower()
                if 'angry' in query1 or 'anger' in query1 or 'annoyed' in query1 or 'irritated' in query1:
                    angry()
                    p=1
                elif 'disgust' in query1 or 'displeased' in query1 or 'unpleasant' in query1:
                    disgust()
                    p=1
                elif 'calm' in query1 or 'neutral' in query1 or 'restful' in query1 or 'peaceful' in query1:
                    neutral()
                    p=1
                elif 'happy' in query1 or 'joyful' in query1 or 'cheerful' in query1 or 'blissful' in query1:
                    happy()
                    p=1
                elif 'fearful' in query1 or 'frightful' in query1 or 'horrible' in query1 or 'fearsome' in query1 or 'dreadful' in query1:
                    fear()
                    p=1
                elif 'surprised' in query1 or 'shocked' in query1 or 'amazed' in query1 or 'staggered' in query1:
                    surprise()
                    p=1
                else:
                    speak("Not fed with such emotion")

            else:
                speak("sorry! didn't get you")

            

        

            
        

        elif 'angry' in emotion_list:
            speak("are you feeling angry or annoyed")
            query=takeCommand().lower()
        
            if 'yes' in query or 'right' in query:
                angry()
                p=1
            elif 'no' in query or 'not' in query or 'wrong' in query:
                speak("please tell me what are you feeling")
                query1=takeCommand().lower()
                if 'sad' in query1 or 'low' in query1 or 'down' in query1 or 'not good' in query1 or 'not well' in query1:
                    sad()
                    p=1
                    
                elif 'disgust' in query1 or 'displeased' in query1 or 'unpleasant' in query1:
                    disgust()
                    p=1
               
                elif 'calm' in query1 or 'neutral' in query1 or 'restful' in query1 or 'peaceful' in query1:
                    neutral()
                    p=1
                    
                elif 'happy' in query1 or 'joyful' in query1 or 'cheerful' in query1 or 'blissful' in query1:
                    happy()
                    p=1
                    
                elif 'fearful' in query1 or 'frightful' in query1 or 'horrible' in query1 or 'fearsome' in query1 or 'dreadful' in query1:
                    fear()
                    p=1
                    
                elif 'surprised' in query1 or 'shocked' in query1 or 'amazed' in query1 or 'staggered' in query1:
                    surprise()
                    p=1
                    
                else:
                    speak("Not fed with such emotion")

            else:
                speak("sorry! didn't get you")





        elif 'disgust' in emotion_list:

            speak("are you feeling disgust or displeased")
            query=takeCommand().lower()
        
            if 'yes' in query or 'right' in query:
                disgust()
                p=1

            elif 'no' in query or 'not' in query or 'wrong' in query:
                speak("please tell me what are you feeling")
                query1=takeCommand().lower()
                if 'angry' in query1 or 'anger' in query1 or 'annoyed' in query1 or 'irritated' in query1:
                    angry()
                    p=1
                    
                elif 'sad' in query1 or 'low' in query1 or 'down' in query1 or 'not good' in query1 or 'not well' in query1:
                    sad()
                    p=1
               
                elif 'calm' in query1 or 'neutral' in query1 or 'restful' in query1 or 'peaceful' in query1:
                    neutral()
                    p=1
                    
                elif 'happy' in query1 or 'joyful' in query1 or 'cheerful' in query1 or 'blissful' in query1:
                    happy()
                    p=1
                    
                elif 'fearful' in query1 or 'frightful' in query1 or 'horrible' in query1 or 'fearsome' in query1 or 'dreadful' in query1:
                    fear()
                    p=1
                    
                elif 'surprised' in query1 or 'shocked' in query1 or 'amazed' in query1 or 'staggered' in query1:
                    surprise()
                    p=1
                    
                else:
                    speak("Not fed with such emotion")

            else:
                speak("sorry! didn't get you")





        elif 'neutral' in emotion_list or 'calm' in emotion_list:
            speak("are you feeling calm or neutral")
            query=takeCommand().lower()
            if 'yes' in query or 'right' in query:
                neutral()
                p=1
            elif 'no' in query or 'not' in query or 'wrong' in query:
                speak("please tell me what are you feeling")
                query1=takeCommand().lower()
                if 'angry' in query1 or 'anger' in query1 or 'annoyed' in query1 or 'irritated' in query1:
                    angry()
                    p=1
                    
                elif 'disgust' in query1 or 'displeased' in query1 or 'unpleasant' in query1:
                    disgust()
                    p=1
               
                elif 'sad' in query1 or 'low' in query1 or 'down' in query1 or 'not good' in query1 or 'not well' in query1:
                    sad()
                    p=1
                    
                elif 'happy' in query1 or 'joyful' in query1 or 'cheerful' in query1 or 'blissful' in query1:
                    happy()
                    p=1
                    
                elif 'fearful' in query1 or 'frightful' in query1 or 'horrible' in query1 or 'fearsome' in query1 or 'dreadful' in query1:
                    fear()
                    p=1
                    
                elif 'surprised' in query1 or 'shocked' in query1 or 'amazed' in query1 or 'staggered' in query1:
                    surprise()
                    p=1
                    
                else:
                    speak("Not fed with such emotion")
            else:
                speak("sorry! didn't get you")






        elif 'happy' in emotion_list:
            speak("are you feeling happy or joyful")
            query=takeCommand().lower()
        
        
            if 'yes' in query or 'right' in query:
                happy()
                p=1
            elif 'no' in query or 'not' in query or 'wrong' in query:
                speak("please tell me what are you feeling")
                query1=takeCommand().lower()
                if 'angry' in query1 or 'anger' in query1 or 'annoyed' in query1 or 'irritated' in query1:
                    angry()
                    p=1
                elif 'disgust' in query1 or 'displeased' in query1 or 'unpleasant' in query1:
                    disgust()
                    p=1
                elif 'sad' in query1 or 'low' in query1 or 'down' in query1 or 'not good' in query1 or 'not well' in query1:
                    sad()
                    p=1
                elif 'calm' in query1 or 'neutral' in query1 or 'restful' in query1 or 'peaceful' in query1:
                    neutral()
                    p=1
                elif 'fearful' in query1 or 'frightful' in query1 or 'horrible' in query1 or 'fearsome' in query1 or 'dreadful' in query1:
                    fear()
                    p=1
                elif 'surprised' in query1 or 'shocked' in query1 or 'amazed' in query1 or 'staggered' in query1:
                    surprise()
                    p=1
                else:
                    speak("Not fed with such emotion")
            else:
                speak("sorry! didn't get you")






        elif 'fear' in emotion_list:
            speak("are you feeling fearful")
            query=takeCommand().lower()
        
            if 'yes' in query or 'right' in query:
                fear()
                p=1
            elif 'no' in query or 'not' in query or 'wrong' in query:
                speak("please tell me what are you feeling")
                query1=takeCommand().lower()
                if 'angry' in query1 or 'anger' in query1 or 'annoyed' in query1 or 'irritated' in query1:
                    angry()
                    p=1
                elif 'disgust' in query1 or 'displeased' in query1 or 'unpleasant' in query1:
                    disgust()
                    p=1
               
                elif 'sad' in query1 or 'low' in query1 or 'down' in query1 or 'not good' in query1 or 'not well' in query1:
                    sad()
                    p=1
                elif 'happy' in query1 or 'joyful' in query1 or 'cheerful' in query1 or 'blissful' in query1:
                    happy()
                    p=1
                elif 'calm' in query1 or 'neutral' in query1 or 'restful' in query1 or 'peaceful' in query1:
                    neutral()
                    p=1
                elif 'surprised' in query1 or 'shocked' in query1 or 'amazed' in query1 or 'staggered' in query1:
                    surprise()
                    p=1
                    
                else:
                    speak("Not fed with such emotion")
            else:
                speak("sorry! didn't get you")





        elif 'surprise' in emotion_list:
            speak("are you feeling surprised or shocked")
            query=takeCommand().lower()
        
            if 'yes' in query or 'right' in query:
                surprise()
                p=1
            elif 'no' in query or 'not' in query or 'wrong' in query:
                speak("please tell me what are you feeling")
                query1=takeCommand().lower()
                if 'angry' in query1 or 'anger' in query1 or 'annoyed' in query1 or 'irritated' in query1:
                    angry()
                    p=1
                elif 'disgust' in query1 or 'displeased' in query1 or 'unpleasant' in query1:
                    disgust()
                    p=1
               
                elif 'sad' in query1 or 'low' in query1 or 'down' in query1 or 'not good' in query1 or 'not well' in query1:
                    sad()
                    p=1
                elif 'happy' in query1 or 'joyful' in query1 or 'cheerful' in query1 or 'blissful' in query1:
                    happy()
                    p=1
                elif 'fearful' in query1 or 'frightful' in query1 or 'horrible' in query1 or 'fearsome' in query1 or 'dreadful' in query1:
                    fear()
                    p=1
                elif 'calm' in query1 or 'neutral' in query1 or 'restful' in query1 or 'peaceful' in query1:
                    neutral()
                    p=1
                else:
                    speak("Not fed with such emotion")
            else:
                speak("sorry! didn't get you")

    speak("Good bye. Have a nice day")

    


