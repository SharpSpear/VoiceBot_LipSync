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
# from streamlit_webrtc import WebRtcMode, webrtc_streamer
import openai
import io
import speech_recognition as sr
import json
import cv2

openai.api_key = "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"


HERE = Path(__file__).parent

source_folder = os.path.join(HERE, 'source')

if not os.path.exists(source_folder):
  os.makedirs(source_folder)

# logger = logging.getLogger(__name__)

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

def speak(text):
    tts = gTTS(text=text, lang="en")
    filename = os.path.join(source_folder, "voice.mp3")
    tts.save(filename)

def main():
    st.set_page_config(
        page_title='Voice ChatBot!',
        page_icon='🤖',
        layout='wide',
        initial_sidebar_state='expanded'
    )        

    st.title('Voice ChatBot!')

    question = ''
    answer = ''
    
    r = sr.Recognizer()
    if st.button('Press and Say!'):
      with sr.Microphone() as resource:
              audio = r.listen(resource)
              try:
                  question = r.recognize_google(audio)
                  # print(said)
              except Exception as e:
                  print("Exception: " + str(e))

    st.sidebar.header('Voice Config')
    voice_type = st.sidebar.radio(
        'Select Voice', ['Male', 'Female']
    )
          
    st.sidebar.header('Image/Video Config')
    source_radio = st.sidebar.radio(
        'Select Source', ['Image', 'Video']
    )

    # question = 'what technology can you use for data mining'
    
    question_field = st.empty()
    question_field.markdown(f"**Question:** {question}")
    
    if not question == '':
      answer = generate_response(question)
      data = {
        'question': question,
        'answer': answer,
      }
      with open('json_data.json', 'w') as outfile:
        json.dump(data, outfile)
    if not answer == '':
      speak(answer)
    
    answer_field = st.empty()
    answer_field.markdown(f"**Answer:** {answer}")

    
    col1, col2 = st.columns(2)
    
    with col1:
      source = None
      if source_radio == 'Image':
        source = st.sidebar.file_uploader(
          "Choose an Image...", type=('jpg', 'png', 'bmp', 'jpeg')
        )
        try:
          if source is None:
            # default_image_path = os.path.join(source_folder, 'default_image.jpg')
            default_image_path = 'source/default_image.jpg'
            # default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption='Default Image', use_column_width=True)
            source = default_image_path
            # uploaded_image = PIL.Image.open(default_image_path)
          else:
            uploaded_image = PIL.Image.open(source)
            uploaded_image.save(os.path.join(source_folder, 'uploaded_image.jpg'))
            st.image(source, caption='Uploaded Image', use_column_width=True)
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
            st.video(default_video_path)
            source = default_video_path
            # uploaded_image = PIL.Image.open(default_image_path)
          else:
            g = io.BytesIO(source.read())  ## BytesIO Object
            temporary_location = os.path.join(source_folder, "uploaded_video.mp4")

            with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file

            # close file
            out.close()
            st.video(temporary_location)
            source = temporary_location
        except Exception as ex:
          st.error('Error occurred while opening the video')
          st.error(ex)
    tmp = question
    if st.sidebar.button('Start the Lip-sync'):
      with open('json_data.json', 'r') as json_file:
        data = json.load(json_file)
      question_field.markdown(f"**Question:** {data['question']}")
      answer_field.markdown(f"**Answer:** {data['answer']}")
      if not os.path.isfile(os.path.join(source_folder, 'voice.mp3')):
        st.sidebar.error('Please press the "Press and Say" button and say your question!')
        return
      output_file_path = os.path.join(source_folder, 'result.mp4')
      if os.path.isfile(output_file_path):
        os.remove(output_file_path)
      try:
        cmd = rf"python inference.py --checkpoint_path checkpoints/wav2lip_1.pth --face {source} --audio source/voice.mp3 --outfile {output_file_path} --nosmooth"
        os.system(cmd)
      except:
        print('Error')
    with col2:
      output_file_path = os.path.join(source_folder, 'result.mp4')
      # vid_area = st.empty()
      # while True:
      #   cap = cv2.VideoCapture(source)
      #   if (cap.isOpened()== False): 
      #     print("Error opening video stream or file")   
      #   while(cap.isOpened()):
      #     ret, frame = cap.read()
      #     if ret == True:      
      #       vid_area.image(frame,channels='BGR')
      #       # cv2.imshow('Frame',frame)
      #       if cv2.waitKey(25) & 0xFF == ord('q'):
      #         break
      #     else: 
      #       break
      if  os.path.isfile(os.path.join(source_folder, 'result.mp4')):
        st.video(os.path.join(source_folder, 'result.mp4'))

if __name__ == "__main__":
    main()







# # This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
# def download_file(url, download_to: Path, expected_size=None):
#     # Don't download the file twice.
#     # (If possible, verify the download using the file length.)
#     if download_to.exists():
#         if expected_size:
#             if download_to.stat().st_size == expected_size:
#                 return
#         else:
#             st.info(f"{url} is already downloaded.")
#             if not st.button("Download again?"):
#                 return

#     download_to.parent.mkdir(parents=True, exist_ok=True)

#     # These are handles to two visual elements to animate.
#     weights_warning, progress_bar = None, None
#     try:
#         weights_warning = st.warning("Downloading %s..." % url)
#         progress_bar = st.progress(0)
#         with open(download_to, "wb") as output_file:
#             with urllib.request.urlopen(url) as response:
#                 length = int(response.info()["Content-Length"])
#                 counter = 0.0
#                 MEGABYTES = 2.0 ** 20.0
#                 while True:
#                     data = response.read(8192)
#                     if not data:
#                         break
#                     counter += len(data)
#                     output_file.write(data)

#                     # We perform animation by overwriting the elements.
#                     weights_warning.warning(
#                         "Downloading %s... (%6.2f/%6.2f MB)"
#                         % (url, counter / MEGABYTES, length / MEGABYTES)
#                     )
#                     progress_bar.progress(min(counter / length, 1.0))
#     # Finally, we remove these visual elements by calling .empty().
#     finally:
#         if weights_warning is not None:
#             weights_warning.empty()
#         if progress_bar is not None:
#             progress_bar.empty()


# # This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
# @st.cache_data  # type: ignore
# def get_ice_servers():
#     """Use Twilio's TURN server because Streamlit Community Cloud has changed
#     its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
#     We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
#     but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
#     See https://github.com/whitphx/streamlit-webrtc/issues/1213
#     """

#     # Ref: https://www.twilio.com/docs/stun-turn/api
#     try:
#         account_sid = os.environ["TWILIO_ACCOUNT_SID"]
#         auth_token = os.environ["TWILIO_AUTH_TOKEN"]
#     except KeyError:
#         logger.warning(
#             "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
#         )
#         return [{"urls": ["stun:stun.l.google.com:19302"]}]

#     client = Client(account_sid, auth_token)

#     token = client.tokens.create()

#     return token.ice_servers




    # https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
    # MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    # LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    # MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
    # LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

    # download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
    # download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

    # lm_alpha = 0.931289039105002
    # lm_beta = 1.1834137581510284
    # beam = 100

    # sound_only_page = "Sound only (sendonly)"
    # with_video_page = "With video (sendrecv)"
    # app_mode = st.selectbox("Choose the app mode", [sound_only_page, with_video_page])

    # if app_mode == sound_only_page:
    #     question = app_sst(
    #         str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
    #     )
    # elif app_mode == with_video_page:
    #     question = app_sst_with_video(
    #         str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
    #     )


    
    
    


# def app_sst(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):
#     webrtc_ctx = webrtc_streamer(
#         key="speech-to-text",
#         mode=WebRtcMode.SENDONLY,
#         audio_receiver_size=1024,
#         rtc_configuration={"iceServers": get_ice_servers()},
#         media_stream_constraints={"video": False, "audio": True},
#     )

#     status_indicator = st.empty()

#     if not webrtc_ctx.state.playing:
#         return

#     status_indicator.write("Loading...")
#     text_output = st.empty()
#     stream = None

#     while True:
#         if webrtc_ctx.audio_receiver:
#             if stream is None:
#                 from deepspeech import Model

#                 model = Model(model_path)
#                 model.enableExternalScorer(lm_path)
#                 model.setScorerAlphaBeta(lm_alpha, lm_beta)
#                 model.setBeamWidth(beam)

#                 stream = model.createStream()

#                 status_indicator.write("Model loaded.")

#             sound_chunk = pydub.AudioSegment.empty()
#             try:
#                 audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#             except queue.Empty:
#                 time.sleep(0.1)
#                 status_indicator.write("No frame arrived.")
#                 continue

#             status_indicator.write("Running. Say something!")

#             for audio_frame in audio_frames:
#                 sound = pydub.AudioSegment(
#                     data=audio_frame.to_ndarray().tobytes(),
#                     sample_width=audio_frame.format.bytes,
#                     frame_rate=audio_frame.sample_rate,
#                     channels=len(audio_frame.layout.channels),
#                 )
#                 sound_chunk += sound

#             if len(sound_chunk) > 0:
#                 sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
#                     model.sampleRate()
#                 )
#                 buffer = np.array(sound_chunk.get_array_of_samples())
#                 stream.feedAudioContent(buffer)
#                 text = stream.intermediateDecode()
#                 text_output.markdown(f"**Text:** {text}")
#         else:
#             status_indicator.write("AudioReciver is not set. Abort.")
#             break
#     return text


# def app_sst_with_video(
#     model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int
# ):
#     frames_deque_lock = threading.Lock()
#     frames_deque: deque = deque([])

#     async def queued_audio_frames_callback(
#         frames: List[av.AudioFrame],
#     ) -> av.AudioFrame:
#         with frames_deque_lock:
#             frames_deque.extend(frames)

#         # Return empty frames to be silent.
#         new_frames = []
#         for frame in frames:
#             input_array = frame.to_ndarray()
#             new_frame = av.AudioFrame.from_ndarray(
#                 np.zeros(input_array.shape, dtype=input_array.dtype),
#                 layout=frame.layout.name,
#             )
#             new_frame.sample_rate = frame.sample_rate
#             new_frames.append(new_frame)

#         return new_frames

#     webrtc_ctx = webrtc_streamer(
#         key="speech-to-text-w-video",
#         mode=WebRtcMode.SENDRECV,
#         queued_audio_frames_callback=queued_audio_frames_callback,
#         rtc_configuration={"iceServers": get_ice_servers()},
#         media_stream_constraints={"video": True, "audio": True},
#     )

#     status_indicator = st.empty()

#     if not webrtc_ctx.state.playing:
#         return

#     status_indicator.write("Loading...")
#     text_output = st.empty()
#     stream = None

#     while True:
#         if webrtc_ctx.state.playing:
#             if stream is None:
#                 from deepspeech import Model

#                 model = Model(model_path)
#                 model.enableExternalScorer(lm_path)
#                 model.setScorerAlphaBeta(lm_alpha, lm_beta)
#                 model.setBeamWidth(beam)

#                 stream = model.createStream()

#                 status_indicator.write("Model loaded.")

#             sound_chunk = pydub.AudioSegment.empty()

#             audio_frames = []
#             with frames_deque_lock:
#                 while len(frames_deque) > 0:
#                     frame = frames_deque.popleft()
#                     audio_frames.append(frame)

#             if len(audio_frames) == 0:
#                 time.sleep(0.1)
#                 status_indicator.write("No frame arrived.")
#                 continue

#             status_indicator.write("Running. Say something!")

#             for audio_frame in audio_frames:
#                 sound = pydub.AudioSegment(
#                     data=audio_frame.to_ndarray().tobytes(),
#                     sample_width=audio_frame.format.bytes,
#                     frame_rate=audio_frame.sample_rate,
#                     channels=len(audio_frame.layout.channels),
#                 )
#                 sound_chunk += sound

#             if len(sound_chunk) > 0:
#                 sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
#                     model.sampleRate()
#                 )
#                 buffer = np.array(sound_chunk.get_array_of_samples())
#                 stream.feedAudioContent(buffer)
#                 text = stream.intermediateDecode()
#                 text_output.markdown(f"**Text:** {text}")
#         else:
#             status_indicator.write("Stopped.")
#             break
#     return text


# if __name__ == "__main__":
#     # import os

#     # DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

#     # logging.basicConfig(
#     #     format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
#     #     "%(message)s",
#     #     force=True,
#     # )

#     # logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

#     # st_webrtc_logger = logging.getLogger("streamlit_webrtc")
#     # st_webrtc_logger.setLevel(logging.DEBUG)

#     # fsevents_logger = logging.getLogger("fsevents")
#     # fsevents_logger.setLevel(logging.WARNING)

#     main()