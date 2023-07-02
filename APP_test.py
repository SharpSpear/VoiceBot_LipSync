import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
import os
from collections import deque
from pathlib import Path
from typing import List
import PIL
import av
import numpy as np
import pydub
import streamlit as st
from twilio.rest import Client
from gtts import gTTS
import openai
import io
import speech_recognition as sr
import json
import cv2
import queue
import threading
import speech_recognition as sr
from streamlit.runtime.scriptrunner import add_script_run_ctx
from APP_test_utils import *
from gfpgan import GFPGANer
from playsound import playsound
import shutil
from ffpyplayer.player import MediaPlayer

openai.api_key = "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"

HERE = Path(__file__).parent

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

source_folder = os.path.join(HERE, 'source')

face_enhancer = GFPGANer(
			model_path='face_enhance.pth',
			upscale=1,
			arch='clean',
			channel_multiplier=2,
			bg_upsampler=None)

model = load_model('checkpoints/wav2lip_1.pth')

if not os.path.exists(source_folder):
  os.makedirs(source_folder)

def generate_response(prompt):
    completions = openai.Completion.create(
        engine = 'text-davinci-003',
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature = 0.5,
    )
    message = completions.choices[0].text
    return message

def generate_result(input_face, input_audio):
  if not os.path.isfile(input_face):
    raise ValueError('--face argument must be a valid path to video/image file')
  elif input_face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    full_frames = [cv2.imread(input_face)]
    fps = 25
  else:
    video_stream = cv2.VideoCapture(input_face)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    print('Reading video frames...')

    full_frames = []
    while 1:
      still_reading, frame = video_stream.read()
      if not still_reading:
        video_stream.release()
        break

      y1, y2, x1, x2 = 0, -1, 0, -1
      if x2 == -1: x2 = frame.shape[1]
      if y2 == -1: y2 = frame.shape[0]

      frame = frame[y1:y2, x1:x2]

      full_frames.append(frame)
  print ("Number of frames available for inference: "+str(len(full_frames)))
  
  if not input_audio.endswith('.wav'):
    print('Extracting raw audio...')
    command = 'ffmpeg -y -i {} -strict -2 {}'.format(input_audio, 'source/voice.wav')
    subprocess.call(command, shell=True)
    input_audio = 'source/voice.wav'
  
  wav = audio.load_wav(input_audio, 16000)
  mel = audio.melspectrogram(wav)
  print(mel.shape)
  
  if np.isnan(mel.reshape(-1)).sum() > 0:
    raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

  mel_chunks = []
  mel_idx_multiplier = 80./fps 
  i = 0
  
  while 1:
    start_idx = int(i * mel_idx_multiplier)
    if start_idx + mel_step_size > len(mel[0]):
      mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
      break
    mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
    i += 1
  print("Length of mel chunks: {}".format(len(mel_chunks)))
  
  full_frames = full_frames[:len(mel_chunks)]
  batch_size = 128
  gen = datagen(full_frames.copy(), mel_chunks)
  
  for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
    if i == 0:
      frame_h, frame_w = full_frames[0].shape[:-1]
      out = cv2.VideoWriter('temp/result.avi', 
                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

    with torch.no_grad():
      pred = model(mel_batch, img_batch)

    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
    kk = 0
    for p, f, c in zip(pred, frames, coords):
      y1, y2, x1, x2 = c
      p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
      print(kk)
      _, _, p = face_enhancer.enhance(p, has_aligned=False, only_center_face=False, paste_back=True)
      f[y1:y2, x1:x2] = p
      out.write(f)
      kk += 1

  out.release()

  command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(input_audio, 'temp/result.avi', 'source/result.mp4')
  subprocess.call(command, shell=True)

 
def speak(text, count):
    tts = gTTS(text=text, lang="en")
    filename = "source/voice.mp3"
    tts.save(filename)
    if os.path.isfile(filename):
      generate_result(source, filename)
      shutil.copy(filename, f'dataset/voice_{count}.mp3')
      os.remove(filename)
      first_data = {
            'question': '',
            'answer': '',
          }
      with open('json_data.json', 'w') as outfile:
        json.dump(first_data, outfile)
    
def voice_to_text(q):
    recognizer = sr.Recognizer()
    with sr.Microphone() as mic:
        while True:
            audio = recognizer.listen(mic)
            try:
                # Recognize speech using Google Speech Recognition
                text = recognizer.recognize_google(audio)
                # Put the result into the queue
                q.put(text)
            except sr.UnknownValueError:
                print("Could not understand audio")
                # pass
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
                # pass
            
def display_text(q):
    count = 0
    while True:
        if q.empty():
          with open('json_data.json', 'r') as json_file:
            data = json.load(json_file)
          question = data['question']
          answer = data['answer']
          # question = ''
          # answer = ''
        # If the queue is not empty
        else:
            # Get the text from the queue and print it
            question = q.get()
            print('question', question) 
        
        question_field.markdown(f"**Question:** {question}")
        if not question == '':
          answer = generate_response(question)
          # answer = 'nice to meet you, Thomas'
          data = {
            'question': question,
            'answer': answer,
          }
          with open('json_data.json', 'w') as outfile:
            json.dump(data, outfile)
        answer_field.markdown(f"**Answer:** {answer}")
        
        
        if answer != '':
            speak(answer, count)
            count += 1      
        
def display_video(q):
  count = 0
  while 1:
    while 1:
      
      if not os.path.isfile('source/result.mp4'): 
        break
      else:
          
        cap = cv2.VideoCapture('source/result.mp4')
        player = MediaPlayer('source/result.mp4')
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            break
        while(cap.isOpened()):
            ret, frame = cap.read()
            print('Test', ret)
            if ret == True:   
              if cv2.waitKey(40) & 0xFF == ord('q'):
                    break   
              vid_area.image(frame,channels='BGR')
                # cv2.imshow('Frame',frame)
                
            else: 
              print('End')
              cap.release()
              break
      shutil.copy('source/result.mp4', f'dataset/result_{count}.mp4')
      player.close_player()
      os.remove('source/result.mp4')
    while 1:
      if os.path.isfile('source/result.mp4'): 
        print('-------------------There is a result.mp4 file in this folder--------------------')
        break
      cap = cv2.VideoCapture(source)
      if (cap.isOpened()== False): 
          print("Error opening video stream or file")
      while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:      
              vid_area.image(frame,channels='BGR')
              # cv2.imshow('Frame',frame)
              if cv2.waitKey(25) & 0xFF == ord('q'):
                  break
          else: 
              break
# def display_audio(q):
#   while 1:
#     if os.path.isfile('source/result.mp4'):
#       playsound('source/voice.mp3') 
#     if os.path.isfile('source/voice.mp3'):
#       os.remove('source/voice.mp3')
  
if __name__ == "__main__":
    first_data = {
            'question': '',
            'answer': '',
          }
    with open('json_data.json', 'w') as outfile:
      json.dump(first_data, outfile)
    st.set_page_config(
        page_title='Voice ChatBot!',
        page_icon='🤖',
        layout='wide',
        initial_sidebar_state='expanded'
    ) 
    st.title('Voice ChatBot!')
    
    st.sidebar.header('Voice Config')
    voice_type = st.sidebar.radio(
        'Select Voice', ['Male', 'Female']
    )
          
    st.sidebar.header('Image/Video Config')
    source_radio = st.sidebar.radio(
        'Select Source', ['Video', 'Image']
    )
    
    question_field = st.empty()
    answer_field = st.empty()
    
    if source_radio == 'Image':
        source = st.sidebar.file_uploader(
          "Choose an Image...", type=('jpg', 'png', 'bmp', 'jpeg')
        )
        try:
          if source is None:
            # default_image_path = os.path.join(source_folder, 'default_image.jpg')
            default_image_path = 'source/default_image.jpg'
            # default_image = PIL.Image.open(default_image_path)
            # st.image(default_image_path, caption='Default Image', use_column_width=True)
            source = default_image_path
            # uploaded_image = PIL.Image.open(default_image_path)
          else:
            uploaded_image = PIL.Image.open(source)
            uploaded_image.save(os.path.join(source_folder, 'uploaded_image.jpg'))
            # st.image(source, caption='Uploaded Image', use_column_width=True)
            source = os.path.join(source_folder, 'uploaded_image.jpg')
        except Exception as ex:
          st.error('Error occurred while opening the image')
          st.error(ex)
    elif source_radio == 'Video':
        source = st.sidebar.file_uploader(
          "Choose an Video...", type=('avi', 'mp4', 'mpeg')
        )
        try:
          if source is None:
            default_video_path = os.path.join(source_folder, 'default_video.mp4')
            # default_image = PIL.Image.open(default_image_path)
            # st.video(default_video_path)
            source = default_video_path
            # uploaded_image = PIL.Image.open(default_image_path)
          else:
            g = io.BytesIO(source.read())  ## BytesIO Object
            temporary_location = os.path.join(source_folder, "uploaded_video.mp4")

            with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file

            # close file
            out.close()
            # st.video(temporary_location)
            source = temporary_location
        except Exception as ex:
          st.error('Error occurred while opening the video')
          st.error(ex)
          
    vid_area = st.empty()
    
    q = queue.Queue()
    
    t1 = threading.Thread(target=voice_to_text, args=(q,))
    t2 = threading.Thread(target=display_text, args=(q,))
    t3 = threading.Thread(target=display_video, args=(q, ))
    # t4 = threading.Thread(target=display_audio, args=(q, ))
    # add_script_run_ctx(t1)
    # if st.sidebar.button('Start the Lip-sync'):
    add_script_run_ctx(t2)
    add_script_run_ctx(t3)
    # add_script_run_ctx(t4)
    t1.start()
    t2.start()
    t3.start()
    # t4.start()

    t1.join()
    t2.join()
    t3.join()
    # t4.join()

