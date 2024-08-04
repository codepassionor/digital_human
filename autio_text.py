import os
import subprocess
import pyaudio
import wave
import speech_recognition as sr
import openai
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from openai import OpenAI
import sys
from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

# from src.utils.preprocess import CropAndExtract
# from src.test_audio2coeff import Audio2Coeff  
# from src.facerender.animate import AnimateFromCoeff
# from src.generate_batch import get_data
# from src.generate_facerender_batch import get_facerender_data
# from src.utils.init_path import init_path

# def main(args):
#     #torch.backends.cudnn.enabled = False

#     pic_path = args.source_image
#     audio_path = args.driven_audio
#     save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
#     os.makedirs(save_dir, exist_ok=True)
#     pose_style = args.pose_style
#     device = args.device
#     batch_size = args.batch_size
#     input_yaw_list = args.input_yaw
#     input_pitch_list = args.input_pitch
#     input_roll_list = args.input_roll
#     ref_eyeblink = args.ref_eyeblink
#     ref_pose = args.ref_pose

#     current_root_path = os.path.split(sys.argv[0])[0]

#     sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

#     #init model
#     preprocess_model = CropAndExtract(sadtalker_paths, device)

#     audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
#     animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

#     #crop image and extract 3dmm from image
#     first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
#     os.makedirs(first_frame_dir, exist_ok=True)
#     print('3DMM Extraction for source image')
#     first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
#                                                                              source_image_flag=True, pic_size=args.size)
#     if first_coeff_path is None:
#         print("Can't get the coeffs of the input")
#         return

#     if ref_eyeblink is not None:
#         ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
#         ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
#         os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
#         print('3DMM Extraction for the reference video providing eye blinking')
#         ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
#     else:
#         ref_eyeblink_coeff_path=None

#     if ref_pose is not None:
#         if ref_pose == ref_eyeblink: 
#             ref_pose_coeff_path = ref_eyeblink_coeff_path
#         else:
#             ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
#             ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
#             os.makedirs(ref_pose_frame_dir, exist_ok=True)
#             print('3DMM Extraction for the reference video providing pose')
#             ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
#     else:
#         ref_pose_coeff_path=None

#     #audio2ceoff
#     batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
#     coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

#     # 3dface render
#     if args.face3dvis:
#         from src.face3d.visualize import gen_composed_video
#         gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
#     #coeff2video
#     data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
#                                 batch_size, input_yaw_list, input_pitch_list, input_roll_list,
#                                 expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
    
#     result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
#                                 enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
    
#     shutil.move(result, save_dir+'.mp4')
#     print('The generated video is named:', save_dir+'.mp4')

#     if not args.verbose:
#         shutil.rmtree(save_dir)



# def play_audio(file_path):
#     # Open the audio file
#     wf = wave.open(file_path, 'rb')

#     # Create an interface to PortAudio
#     p = pyaudio.PyAudio()

#     # Open a .Stream object to write the WAV file to
#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                     channels=wf.getnchannels(),
#                     rate=wf.getframerate(),
#                     output=True)

#     # Read data in chunks
#     chunk = 1024
#     data = wf.readframes(chunk)

#     # Play the sound by writing the audio data to the stream
#     while data:
#         stream.write(data)
#         data = wf.readframes(chunk)

#     # Close and terminate everything properly
#     stream.close()
#     p.terminate()
    
# Set the API key and base URL globally for the OpenAI client
api_key = "sk-xmY83ICHAbw95y8RBeF0B92143154dF78b0bC762912d78D8"
base_url = "https://openkey.cloud/v1"
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(base_url=base_url, api_key=api_key)

def play_audio(file_path):
    # Load the audio file
    wave_obj = sr.WaveObject.from_wave_file(file_path)
    # Play the audio file
    play_obj = wave_obj.play()
    # Wait for the audio file to finish playing
    play_obj.wait_done()

def generate_speech(answer, output_file, model="tts-1", voice="alloy"):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=answer
    )
    with open(output_file, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

def analyze_text_with_openai(transcription):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a trusted friend of mine, always ready to answer my questions and provide support whenever I need it."},
            {"role": "user", "content": f"{transcription}"}
        ]
    )
    return completion.choices[0].message.content

def record_audio_until_silence(filename, max_record_duration=2, silence_duration=1, sample_format=pyaudio.paInt16, channels=1, fs=44100):
    chunk = 1024
    p = pyaudio.PyAudio()
    print('Recording')
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    frames = []
    silence_threshold = 1
    silence_chunks = 0
    max_silence_chunks = int(silence_duration * fs / chunk)
    max_chunks = int(max_record_duration * fs / chunk)
    for i in range(max_chunks):
        data = stream.read(chunk)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.max(audio_data) < silence_threshold:
            silence_chunks += 1
        else:
            silence_chunks = 0
        if silence_chunks > max_silence_chunks:
            break
    print('Finished recording')
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

def audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcription: " + text)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return ""

def split_audio_on_silence(filename, min_silence_len=10, silence_thresh=-40):
    audio = AudioSegment.from_wav(filename)
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return chunks

# def store_generated_audio(audio_path, source_image, result_dir):
#     command = [
#         'python', 'inference.py',
#         '--driven_audio', audio_path,
#         '--source_image', source_image,
#         '--result_dir', result_dir,
#         '--enhancer', 'gfpgan'
#     ]
#     subprocess.run(command, capture_output=True, text=True)

if __name__ == '__main__':
    count = 0
    if not os.path.exists('checkpoints'):
        subprocess.run(['bash', 'scripts/download_models.sh'])
    
    while True:
        count += 1
        os.makedirs(f"output_{count}", exist_ok=True)
        # Capture audio and save as wav file
        record_audio_until_silence(f"output_{count}/output_{count}.wav", silence_duration=1)

        # Convert audio to text
        transcription = audio_to_text(f"output_{count}/output_{count}.wav")
        if transcription:
            # Analyze transcription with OpenAI
            answer = analyze_text_with_openai(transcription)
            print('---------------------------------------------')
            print(answer)
            print('---------------------------------------------')
            
            # Convert analysis result to audio
            audio_output_path = f"output_{count}/output_{count}_response.wav"
            generate_speech(answer, audio_output_path)
            print(f'Audio saved to {audio_output_path}')
            