from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from concurrent.futures import ThreadPoolExecutor
import torch, os, json, wave, gc
from pydub import AudioSegment
import yt_dlp as youtube_dl

gc.collect()
torch.cuda.empty_cache()

MODEL_INFO = {
    'english': {
        'large': {
            'model_name': 'facebook/wav2vec2-large-960h',
            'size': '1.3 GB'
        },
        'medium': {
            'model_name': 'facebook/wav2vec2-base-960h',
            'size': '380 MB'
        },
        'small': {
            'model_name': 'facebook/wav2vec2-small-960h',
            'size': '100 MB'
        }
    },
    'arabic': {
        'large': {
            'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
            'size': '1.3 GB'
        },
        'medium': {
            'model_name': 'arbml/wav2vec2-large-xlsr-53-arabic-egyptian',
            'size': '1.3 GB'
        },
        'small': {
            'model_name': 'asr-voice/wav2vec2-small-arabic',
            'size': '100 MB'
        }
    }
}

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

def convert_to_mono_wav(input_file):
    sound = AudioSegment.from_file(input_file)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    output_path = os.path.splitext(input_file)[0] + ".wav"
    sound.export(output_path, format="wav")
    return output_path

def transcribe_chunk(processor, model, device, data, framerate):
    # Convert bytes data to numpy array and then to tensor
    waveform = torch.tensor(list(data), dtype=torch.float32).unsqueeze(0)
    waveform = waveform / (2**15)  # Convert to float between -1 and 1
    
    # Ensure the waveform is 1D
    waveform = waveform.squeeze(0)
    
    # Process the input waveform with the correct sampling rate
    input_values = processor(waveform, return_tensors="pt", sampling_rate=framerate).input_values
    
    # Move input values to GPU if available
    input_values = input_values.to(device)
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode predictions
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def transcribe_audio(file_path, model, processor, device, num_threads=4):
    print("Converting audio to WAV format...")
    convert_path = convert_to_mono_wav(file_path)
    
    # Open audio file
    wf = wave.open(convert_path, "rb")

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 48000]:
        raise ValueError("Audio file must be WAV format mono PCM.")

    framerate = wf.getframerate()
    audio_length = wf.getnframes()
    chunk_size = max(audio_length // (num_threads * 2), 1)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            wf.setpos(i * chunk_size)
            data = wf.readframes(chunk_size)
            futures.append(executor.submit(transcribe_chunk, processor, model, device, data, framerate))

        results = [future.result() for future in futures]

    wf.close()

    combined_transcription = " ".join(results)
    return combined_transcription.strip()

def check_audio_format(file_path):
    audio = AudioSegment.from_file(file_path)
    if audio.channels != 1 or audio.frame_rate not in [8000, 16000, 32000, 48000]:
        raise ValueError(
            "Audio file must be WAV format mono PCM with a sample rate of 8000, 16000, 32000, or 48000 Hz.")
    print("Audio file format is correct.")

def load_model_and_processor(model_name):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return processor, model, device

def display_model_info(language):
    if language not in MODEL_INFO:
        raise ValueError(f"Unsupported language: {language}")

    print(f"\nAvailable models for {language.capitalize()}:")
    for size, info in MODEL_INFO[language].items():
        print(f"{size.capitalize()} model: {info['model_name']} - Size: {info['size']}")

def main(video_url, language, model_size, num_threads):
    if language not in MODEL_INFO:
        raise ValueError(f"Unsupported language: {language}")
    if model_size not in MODEL_INFO[language]:
        raise ValueError(f"Unsupported model size: {model_size}")

    model_info = MODEL_INFO[language][model_size]
    model_name = model_info['model_name']

    print("Downloading audio from YouTube...")
    audio_file = download_audio_from_youtube(video_url)

    print("Loading model and processor...")
    processor, model, device = load_model_and_processor(model_name)

    print("Transcribing audio...")
    transcription = transcribe_audio(audio_file, model, processor, device, num_threads)
    print("Transcription completed.")
    print(transcription)
    
    transcription_data = {
        'transcription': transcription
    }
    
    json_file_path = 'transcription.json'
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(transcription_data, json_file, indent=4, ensure_ascii=False)

    print(f"Transcription completed and saved to {json_file_path}.")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    language = input("Select language (English or Arabic): ").lower()
    if language not in ['english', 'arabic']:
        raise ValueError(
            "Invalid language choice. Please choose 'English' or 'Arabic'.")

    display_model_info(language)

    model_size = input("Select model size (large, medium, small): ").lower()
    if model_size not in ['large', 'medium', 'small']:
        raise ValueError(
            "Invalid model size choice. Please choose 'large', 'medium', or 'small'.")

    max_threads = os.cpu_count()
    num_threads = int(input(f"Enter the number of threads (1 to {max_threads}): "))
    if num_threads < 1 or num_threads > max_threads:
        num_threads = max_threads
        
    video_url = input("Enter YouTube video URL: ")
    main(video_url, language, model_size, num_threads)
