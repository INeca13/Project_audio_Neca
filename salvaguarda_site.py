# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:39:35 2022

@author: igorn
"""
#pip install mpld3
import io
import paho.mqtt.client as mqtt
#from random import randrange, uniform
import time
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import mpld3
import streamlit.components.v1 as components
#tratamento
#import pyaudio
#import wave
import os
import hashlib
import numpy as np
import librosa
import librosa.display
from PIL import Image
#import scipy as sp  #se não der, colocar from scipy.fft import fft

mqttBroker = "test.mosquitto.org"
# #"mqtt.eclipseprojects.io"
client = mqtt.Client("Temperature_Inside")
client.connect(mqttBroker)
#NOTA: CASO O LIBROSA DÊ ERRO, FAÇA sudo apt-get install libsndfile1

#centrar imagem inicial
image2 = Image.open('wizard.jpg')
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write(' ')

with col2:
    st.image(image2, caption='https://www.vectorstock.com/royalty-free-vector/wizard-logo-character-design-vector-20749635', width= 450)

with col3:
    st.write(' ')

with col4:
    st.write(' ')

st.title("Audio Feature Extraction Wyzard")




#funções para as features
def amplitude_envelope(signal, frame_size, hop_length):
    """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
    amplitude_envelope = []
    
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
        amplitude_envelope.append(amplitude_envelope_current_frame)
    
    return np.array(amplitude_envelope)    


def fancy_amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])






def plot_magnitude_spectrum(signal, sr, title, f_ratio=1):
    X = np.fft.fft(audio)
    X_mag = np.absolute(X)
    #st.header("Fourier só que com np, e mai fancy com o bins. Talvez usar este")
    plt.figure(figsize=(18, 5))
    
    f = np.linspace(0, sr, len(X_mag)) #temos a freq até à nossa freq de aquisição
    f_bins = int(len(X_mag)*f_ratio) #esta linha serve para mostrar só as freq que estão de acordo com o teoream de nyquist  
    fig3 = plt.figure()
    plt.plot(f[:f_bins], X_mag[:f_bins])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    st.pyplot(fig3)




def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    #st.header("Espetrograma")
    fig7 = plt.figure()
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.xlabel('Tempo')
    plt.ylabel('Amplitude')
    plt.colorbar(format="%+2.f")
    st.pyplot(fig7)



#imagem de sidebar
image = Image.open('imagem.png')
st.sidebar.image(image, caption='Logo FCT/UNL')


st.sidebar.header("Feature extration wizard")
epochs_num = st.sidebar.slider("Tempo /s", 1, 10, key = int) #escolher tempo de recolha de dados
if st.sidebar.button("Start recording"):
    st.sidebar.write(epochs_num)
    #t = time.time()
    client.publish("AAIB/teste/Neca", epochs_num)
    cronometro = time.time()
    st.write('Recording in progress') #informação de quando está a gravar
    while time.time() - cronometro < epochs_num+0.001: 
        if int(time.time() - cronometro+0.5) == int(epochs_num): #acrescentei +0.5 para tentar compensar delay
            st.write('Wait while the file is being received') #mostrar mensagem conforme o ficheri está a ser subscrito
    exec(open('receive-file.py').read())
    st.write('File received!')
    #subscrever()
    


    
    #fazer o calculo das variáveis necessárias para as features
button30 = st.sidebar.checkbox("Ler dados")
if button30:   
    #calculo de parametros para as features
    df5 = pd.read_pickle("copy2-my_data_audio.pkl")
    df6 = df5.to_numpy()
    audio = df6.reshape((len(df6),))



    sr = 22050
    t_audio = len(df6)/sr
    times = np.linspace(0, t_audio, num=len(df6)) #desde 0 a 3, fazer num pontos

    # duration in seconds of 1 sample
    sample_duration = 1 / sr

    # total number of samples in audio file
    tot_samples = len(audio)

    # duration of debussy audio in seconds
    duration = 1 / sr * tot_samples

    FRAME_SIZE = 512
    HOP_LENGTH = 256
    #fourier
    frames = range(len(amplitude_envelope(audio, FRAME_SIZE, HOP_LENGTH)))
    t_1 = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
    X = np.fft.fft(audio)
    #calculo do short time fourier transform
    S_audio = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
        
    Y_audio = np.abs(S_audio) ** 2
    Y_log_audio = librosa.power_to_db(Y_audio)
    rms_audio = librosa.feature.rms(audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    frames_rmse = range(len(rms_audio))
    t_rmse = librosa.frames_to_time(frames_rmse, hop_length=HOP_LENGTH)


    st.sidebar.header("Escolha as features que quer ver") 
    st.sidebar.write("Nota: quanto mais features escolher, mais tempo demora a processar") 
        
    button1 = st.sidebar.checkbox("Audio")
    if button1:
        st.header("Gráfico estático")
        fig1 = plt.figure() 
        som_plot = plt.plot(times, audio)
        plt.xlabel('Tempo')
        plt.ylabel('Amplitude')
        st.pyplot(fig1)
        fn = 'som_plot.png'
        plt.savefig(fn)
        with open(fn, "rb") as img: #guardar gráfico como imagem
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )
        with io.BytesIO() as buffer:#guardar gráfico com ficheiro csv
            # Write array to buffer
            df5 = pd.read_pickle("copy2-my_data_audio.pkl")
            df6 = df5.to_numpy()
            audio = df6.reshape((len(df6),))
            np.savetxt(buffer, audio, delimiter=",")
            st.sidebar.download_button(
                label="Download audio como CSV",
                data = buffer, # Download buffer
                file_name = 'dados_audio.csv',
                mime='text/csv'
        ) 



    button2 = st.sidebar.checkbox("Envelope")
    if button2:
        st.header("Envelope")
        fig2 = plt.figure() 
        librosa.display.waveshow(audio, alpha=0.6)
        amp_env = amplitude_envelope(audio, FRAME_SIZE, HOP_LENGTH)
        envelope_plot = plt.plot(t_1, amplitude_envelope(audio, FRAME_SIZE, HOP_LENGTH), color="r")
        plt.xlabel('Tempo')
        plt.ylabel('Amplitude')
        st.pyplot(fig2)
        fn = 'envelope_plot.png'
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )    

        with io.BytesIO() as buffer:
            # Write array to buffer
            np.savetxt(buffer,  np.column_stack((t_1, amp_env)), delimiter=",")
            st.sidebar.download_button(
                label="Download envelope como CSV",
                data = buffer, # Download buffer
                file_name = 'Envelope_audio.csv',
                mime='text/csv'
            ) 

    button3 = st.sidebar.checkbox("Fourier")
    if button3:
        st.header("Fourier")
        Fourier_plot = plot_magnitude_spectrum(audio, sr, "Gravação", 0.1)
        fn = 'Fourier.png'
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )
        with io.BytesIO() as buffer:
            # Write array to buffer
            X = np.fft.fft(audio)
            X_mag = np.absolute(X)          
            f = np.linspace(0, sr, len(X_mag)) #temos a freq até à nossa freq de aquisição
            f_bins = int(len(X_mag))
            np.savetxt(buffer, np.column_stack((f[:f_bins], X_mag[:f_bins])),  delimiter=",")
            st.sidebar.download_button(
                label="Download Fourier como CSV",
                data = buffer, # Download buffer
                file_name = 'Fourier_audio.csv',
                mime='text/csv'
            )    
        
    button4 = st.sidebar.checkbox("Espetrograma")
    if button4:
        st.header("Espetrograma")
        Espetrograma_plot = plot_spectrogram(Y_audio, sr, HOP_LENGTH)
        fn = 'Espetrograma_plot.png'
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )
        
    button5 = st.sidebar.checkbox("Espetrograma com y em log")
    if button5:
        st.header("Mudança para escala logaritmica de y")
        Espetrograma_ylog_plot = plot_spectrogram(Y_log_audio, sr, HOP_LENGTH)
        fn = 'Espetrograma_ylog_plot.png'
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )

    button6 = st.sidebar.checkbox("Espetrograma com escala log ")
    if button6:
        st.header("Mudança para escala logaritmica")
        Espetrograma_log_plot = plot_spectrogram(Y_log_audio, sr, HOP_LENGTH, y_axis="log")
        fn = 'Espetrograma_log_plot.png'
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )


    button7 = st.sidebar.checkbox("RMSE")
    if button7:
        st.header("RMS Energy")
        fig9 = plt.figure()
        librosa.display.waveshow(audio, alpha=0.6)
        RMSE_plot = plt.plot(t_rmse, rms_audio, color="r")
        #plt.ylim((-0.2, 0.2))
        #plt.title("RMS Energy")
        plt.xlabel('Tempo')
        plt.ylabel('Amplitude')
        st.pyplot(fig9)
        fn = 'RMSE_plot.png'
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )
        with io.BytesIO() as buffer:
            # Write array to buffer
            np.savetxt(buffer,  np.column_stack((t_rmse, rms_audio)), delimiter=",")
            st.sidebar.download_button(
                label="Download envelope como CSV",
                data = buffer, # Download buffer
                file_name = 'Envelope_audio.csv',
                mime='text/csv'
            ) 



    button8 = st.sidebar.checkbox("Dinâmico")
    if button8:
        st.header("gráfico dinâmico")
        fig10 = plt.figure() 
        plt.plot(times, audio)
        fig_html = mpld3.fig_to_html(fig10)
        components.html(fig_html, height=600) 


