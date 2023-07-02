import cv2
import numpy as np
#ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer
from playsound import playsound


video_path="1.mp4"
# playsound('1.mp3')
# print('1111')
def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(40) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()
PlayVideo(video_path)


# from pydub import AudioSegment
# from pydub.playback import play
# sound = AudioSegment.from_file('1.mp3')
# sound = sound.speedup(playback_speed=0.5)
# sound.export("slow.mp3", format="mp3")
# play(sound)
# print('ssss')
# # def speed_change(sound, speed=1.0):
# #     # Manually override the frame_rate. This tells the computer how many
# #     # samples to play per second
# #     sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
# #          "frame_rate": int(sound.frame_rate * speed)
# #       })
# #      # convert the sound with altered frame rate to a standard frame rate
# #      # so that regular playback programs will work right. They often only
# #      # know how to play audio at standard frame rate (like 44.1k)
# #     return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

# # slow_sound = speed_change(sound, 0.75)
# # fast_sound = speed_change(sound, 2.0)

# # 
