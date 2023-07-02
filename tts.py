import pyttsx3
import wave

def speak_with_same_voice(audio_file, input_sentence):
    # Load the audio file into a wave object.
    wave_object = wave.open(audio_file, 'rb')

    # Get the sampling rate of the audio file.
    sampling_rate = wave_object.getframerate()

    # Create a pyttsx3 engine and set the voice to the same voice as the audio file.
    engine = pyttsx3.init()
    voice = engine.getProperty('voices')[0]
    engine.setProperty('voice', voice)

    # Speak the input sentence using the engine.
    engine.say(input_sentence)

    # Save the output audio file.
    output_file = 'output.wav'
    engine.save_to_file(input_sentence, output_file)

if __name__ == '__main__':
    audio_file = 'temp/temp.wav'
    input_sentence = 'This is a test sentence.'

    speak_with_same_voice(audio_file, input_sentence)