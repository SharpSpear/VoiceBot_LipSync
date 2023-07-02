import cv2
import audio

video_file_path = "./test.mp4"
video_stream = cv2.VideoCapture(video_file_path)
fps = video_stream.get(cv2.CAP_PROP_FPS)
full_frames = []
while 1:
    still_reading, frame = video_stream.read()
    if not still_reading:
        video_stream.release()
        break
    y1, y2, x1, x2 = [0, -1, 0, -1]
    if x2 == -1: x2 = frame.shape[1]
    if y2 == -1: y2 = frame.shape[0]
    frame = frame[y1:y2, x1:x2]
    full_frames.append(frame)

audio_path = "./audio.mp4"