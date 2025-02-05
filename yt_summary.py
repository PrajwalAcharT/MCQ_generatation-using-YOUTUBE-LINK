import streamlit as st
from pytube import YouTube
import whisper
from llama_cpp import Llama
import os
import tempfile
import subprocess

def download_audio(url):
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "audio.mp3")
        
        command = [
            "yt-dlp",
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "mp3",
            "--output", output_path,
            url
        ]
        
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            st.error(f"Error downloading video: {result.stderr.decode()}")
            return None
        
        return output_path
    
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

def transcribe_audio(whisper_model, audio_path):
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def generate_summary(llama_model, text):
    prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant that creates clear and concise summaries.
<</SYS>>

Please provide a brief summary of the following video transcript in 2-3 sentences:

{text}

Focus on the main points and key takeaways. [/INST]"""
    
    try:
        response = llama_model(
            prompt,
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            stop=["</s>", "[/INST]", "[INST]"],
            echo=False
        )
        
        summary = response["choices"][0]["text"].strip()
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def load_youtube_models():
    try:
        whisper_model = whisper.load_model("base")
        
        # Update this path to your Llama model 
        model_path = r"C:\Users\91636\Desktop\YouTube-Video-Summarization-App-main\YouTube-Video-Summarization-App-main\myenv\llama-2-7b-32k-instruct.Q4_K_S.gguf"
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None, None
            
        llama_model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_batch=512,
            verbose=False
        )
        return whisper_model, llama_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None