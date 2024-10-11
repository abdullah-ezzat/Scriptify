from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import yt_dlp as youtube_dl
import tensorflow as tf
import os, json

def download_audio_from_youtube(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s',
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    
    return 'audio.wav'

def preprocess_audio(file_path):
    audio_data = tf.io.read_file(file_path)
    waveform, sample_rate = tf.audio.decode_wav(audio_data, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)
    waveform = waveform / tf.reduce_max(tf.abs(waveform))
    waveform = tf.expand_dims(waveform, axis=-1)
    waveform_encoded = tf.audio.encode_wav(waveform, sample_rate)
    return waveform_encoded.numpy()

def convert_to_mono_wav(input_file):
    sound = AudioSegment.from_file(input_file)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    output_path = os.path.splitext(input_file)[0] + ".wav"
    sound.export(output_path, format="wav")
    return output_path


def transcribe_audio(file_path, language):
    model_path = f'models/{language}'
    if not os.path.exists(model_path):
        raise ValueError(f"Model for language '{language}' not found in 'models' directory.")
    
    converted_audio_path = convert_to_mono_wav(file_path)
    model = Model(model_path)
    
    with open(converted_audio_path, "rb") as wf:
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)

        results = []
        while True:
            data = wf.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                results.append(part_result)
        
        final_result = json.loads(rec.FinalResult())
        results.append(final_result)
    
    wf.close()
    
    transcription_text = ""
    word_details = []
    
    for result in results:
        if 'result' in result:
            for word_info in result['result']:
                word = word_info['word']
                start = word_info['start']
                end = word_info['end']
                transcription_text += word + " "
                word_details.append({'word': word, 'start': start, 'end': end})

    transcription_file_path = 'transcription.json'
    with open(transcription_file_path, 'w', encoding='utf-8') as f:
        json.dump({'transcription': transcription_text.strip()}, f, ensure_ascii=False, indent=4)
    
    word_details_file_path = 'word_details.json'
    with open(word_details_file_path, 'w', encoding='utf-8') as f:
        json.dump(word_details, f, ensure_ascii=False, indent=4)

    return json.dumps(word_details, ensure_ascii=False)

def get_folder_name():
    print("Select Model:")
    print("1 - Arabic - Small")
    print("2 - Arabic - Large")
    print("3 - English - Small")
    print("4 - English - Large")
    
    choice = input("Enter the number corresponding to your choice: ")
    folder_names = {
        "1": "arabic-small",
        "2": "arabic-large",
        "3": "english-small",
        "4": "english-large"
    }
    
    return folder_names.get(choice, None)

def main(video_url, language):
    print("Downloading audio from YouTube...")
    audio_file = download_audio_from_youtube(video_url)

    print("Transcribing audio...")
    try:
        transcribe_audio(audio_file, language)
        print("Transcription completed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        os.remove(audio_file)

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    folder_name = get_folder_name()
    if folder_name is None:
        print("Invalid choice. Exiting.")
    else:
        main(video_url, folder_name)