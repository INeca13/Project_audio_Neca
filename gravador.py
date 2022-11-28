# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:07:02 2022

@author: igorn
""" 

import paho.mqtt.client as mqtt
import time
#import matplotlib.pyplot as plt
#import paho.mqtt.client as mqtt
import pandas as pd
#from scipy.fft import fft, fftfreq
import pyaudio
import wave
import numpy as np
import librosa
#import librosa.display
import hashlib
#import IPython.display as ipd

temporizador = time.time()

def send_file():
    broker="test.mosquitto.org"
    filename="my_data_audio.pkl"
    topic="AAIB/audio/Neca"
    qos=1
    data_block_size=2000
    fo=open(filename,"rb")
    file_out="copy2-"+filename
    fout=open(file_out,"wb") #use a different filename
    # for outfile as I'm running sender and receiver together
    
    def process_message(msg):
       """ This is the main receiver code
       """
       print("received ")
       if len(msg)==200: #is header or end
          msg_in=msg.decode("utf-8","ignore")
          msg_in=msg_in.split(",,")
          if msg_in[0]=="end": #is it really last packet?
             in_hash_final=in_hash_md5.hexdigest()
             if in_hash_final==msg_in[2]:
                print("File copied OK -valid hash  ",in_hash_final)
             else:
                print("Bad file receive   ",in_hash_final)
             return False
          else:
             if msg_in[0]!="header":
                in_hash_md5.update(msg)
                return True
             else:
                return False
       else:
          in_hash_md5.update(msg)
          #msg_in=msg.decode("utf-8","ignore")
          if len(msg) <100:
             print(msg)
          return True
    #define callback
    def on_message(client, userdata, message):
       #time.sleep(1)
       #print("received message =",str(message.payload.decode("utf-8")))
       if process_message(message.payload):
          fout.write(message.payload)
    def on_publish(client, userdata, mid):
        #logging.debug("pub ack "+ str(mid))
        client.mid_value=mid
        client.puback_flag=True  
    
    ## waitfor loop
    def wait_for(client,msgType,period=0.005,wait_time=40,running_loop=False):
        client.running_loop=running_loop #if using external loop
        wcount=0
        #return True
        while True:
            #print("waiting"+ msgType)
            if msgType=="PUBACK":
                if client.on_publish:        
                    if client.puback_flag:
                        return True
         
            if not client.running_loop:
                client.loop(.001)  #check for messages manually
            time.sleep(period)
            #print("loop flag ",client.running_loop)
            wcount+=1
            if wcount>wait_time:
                print("return from wait loop taken too long")
                return False
        return True 
    def send_header(filename):
       header="header"+",,"+filename+",,"
       header=bytearray(header,"utf-8")
       header.extend(b','*(200-len(header)))
       print(header)
       c_publish(client,topic,header,qos)
    def send_end(filename):
       end="end"+",,"+filename+",,"+out_hash_md5.hexdigest()
       end=bytearray(end,"utf-8")
       end.extend(b','*(200-len(end)))
       print(end)
       c_publish(client,topic,end,qos)
    def c_publish(client,topic,out_message,qos):
       res,mid=client.publish(topic,out_message,qos)#publish
       #return
    
       if res==0: #published ok
          if wait_for(client,"PUBACK",running_loop=True):
             if mid==client.mid_value:
                print("match mid ",str(mid))
                client.puback_flag=False #reset flag
             else:
                print("quitting")
                raise SystemExit("not got correct puback mid so quitting")
             
          else:
             raise SystemExit("not got puback so quitting")
    
    client= mqtt.Client("client-001")  #create client object client1.on_publish = on_publish                          #assign function to callback client1.connect(broker,port)                                 #establish connection client1.publish("data/files","on")  
    ######
    client.on_message=on_message
    client.on_publish=on_publish
    client.puback_flag=False #use flag in publish ack
    client.mid_value=None
    #####
    print("connecting to broker ",broker)
    client.connect(broker)#connect
    client.loop_start() #start loop to process received messages
    print("subscribing ")
    client.subscribe(topic)#subscribe
    time.sleep(2)
    start=time.time()
    print("publishing ")
    send_header(filename)
    Run_flag=True
    count=0
    out_hash_md5 = hashlib.md5()
    in_hash_md5 = hashlib.md5()
    bytes_out=0
    
    while Run_flag:
       chunk=fo.read(data_block_size)
       if chunk:
          out_hash_md5.update(chunk) #update hash
          out_message=chunk
          #print(" length =",type(out_message))
          bytes_out=bytes_out+len(out_message)
    
          c_publish(client,topic,out_message,qos)
    
             
       else:
          #end of file so send hash
          out_message=out_hash_md5.hexdigest()
          send_end(filename)
          #print("out Message ",out_message)
          #res,mid=client.publish("data/files",out_message,qos=1)#publish
          Run_flag=False
    time_taken=time.time()-start
    print("took ",time_taken)
    print("bytes sent =",bytes_out)
    time.sleep(5)
    client.disconnect() #disconnect
    client.loop_stop() #stop loop
    fout.close()
    fo.close()
        





def audio_ficheiro_txt(RECORD_SECONDS):
    
    
    
    #mqttBroker = "test.mosquitto.org"
    #client = mqtt.Client("Temperature_Inside")
    #client.connect(mqttBroker)
    
    
    
    #NOTA: VER SE ENVIO .WAV OU PICKLE
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    #RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("* recording")
    
    frames = []
    valores =[]
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        data_final = np.frombuffer(data, dtype=np.int32)
        #s=""
        #for i in data_final:
            #s+=str(i)+","
    
        #client.publish("AAIB/test", " ".join(s))
        #valores.append( " ".join(s))
        valores.append(data_final)
    
    print("* done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    
    
    
    #abaixo, temos uma maneira de converter para um array de in16 o sinal de audio
    #waveFile = wave_open('output.wav','rb')
    #nframes = waveFile.getnframes()
    #wavFrames = waveFile.readframes(nframes)
    #ys = np.fromstring(wavFrames, dtype=np.int16) #temos o sinal mais bonito
    
    audio_file = "output.wav"
    audio, sr = librosa.load(audio_file)
    
    #plt.plot(valores)
    #plt.plot(audio)
    
    
    #aqui é basicamente uma maneira de enviar e receber dados
    #os 
    
    # fazer funções e fazer com que cada mqtt dê a respetiva variável
    #df = pd.DataFrame(valores)
    df4 = pd.DataFrame(audio)
    
    
    df4.to_pickle("my_data_audio.pkl")
    time.sleep(2)
    #correr o send-file e recieve-fil
    #exec(open("send-file.py").read())
    
    send_file()
    
    ########################################
    #passar esta aprte para o gitpod
    #df5 = pd.read_pickle("copy-my_data_audio.pkl")
    #df6 = df5.to_numpy()
    #df6 = df6.reshape((len(df6),))
    #plt.plot(df6)
    ########################################
    
    #df.to_pickle("my_data.pkl")
    #exec(open("recieve-file.py").read())
        #df2 = pd.read_pickle("copy-my_data.pkl")
    #df3 = df2.to_numpy()
      
    
    

    


def on_message(client, userdata, message):
    x.append(str(message.payload.decode("utf-8")))
    #e = list(map(float, x))
    #print(int(time.time()-temporizador))
    print("Received message: ", str(message.payload.decode("utf-8")))
    r = list(map(float, x)) #lista de string para float
    if r:
        #r.pop(0)
        RECORD_SECONDS = int(r[-1])
    else:
        r = []
        RECORD_SECONDS = []
            
    if RECORD_SECONDS:
        tempo_record = RECORD_SECONDS
        r = []
        RECORD_SECONDS = []    
        audio_ficheiro_txt(tempo_record)


mqttBroker = "test.mosquitto.org"
# #"mqtt.eclipseprojects.io"
client = mqtt.Client("Smartphone")
client.connect(mqttBroker)


#metemos um while loop para que o programa esteja sempre a correr caso receba commandos
#com frequencia
#fim = 2*60
#while time.time()-temporizador < fim:
    #print(int(time.time()-temporizador))
    #client.loop_start()
x = []
    #print(time.time()-temporizador)
client.subscribe("AAIB/teste/Neca")
client.on_message = on_message
    #time.sleep(3)
    #client.loop_stop()


    

client.loop_forever()
#talvez, com isto podemos fornecer os parametros de gravação


#ver se estamos a receber valore de record
#se sim, ler e guardar o valor e
#se record != [] , chamar a função de audio_ficheiro_txt(record = subscribe)
#por fim,, record = []



#subscribe([("my/topic", 0), ("another/topic", 2)])

###############################################################################


#client.subscribe("RECORD_SECONDS")
#client.on_message = on_message
#client.subscribe("CHUNK")
#client.on_message = on_message2
#time.sleep(4)
#client.loop_stop()

#def on_message(client, userdata, message):
 #   x.append(str(message.payload.decode("utf-8")))
  #  e = list(map(float, x))
   # print("Received message: ", e)
#r = list(map(float, x)) #lista de string para float
#r.pop(0) #o primeiro não conta
#RECORD_SECONDS = r[-1]
