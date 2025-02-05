import streamlit as st
import os
import tempfile
from pytube import YouTube
import whisper
from llama_cpp import Llama
import subprocess
import PyPDF2
from fpdf import FPDF
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Paths (adjust as needed)
MODEL_PATH = r"C:\Users\91636\Desktop\YouTube-Video-Summarization-App-main\YouTube-Video-Summarization-App-main\myenv\LaMini-T5-738M"
LLAMA_MODEL_PATH = r"C:\Users\91636\Desktop\YouTube-Video-Summarization-App-main\YouTube-Video-Summarization-App-main\myenv\llama-2-7b-32k-instruct.Q4_K_S.gguf"

# Add at top of app.py
import torch
import gc

def optimize_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@st.cache_resource
def load_models():
    try:
        whisper_model = whisper.load_model("base")
        
        llama_model = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=4096,  # Increased context
            n_threads=6,  # Adjust based on CPU
            n_batch=256,  # Reduced batch size
            verbose=False
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        mcq_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            device_map='auto',
            torch_dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        return whisper_model, llama_model, tokenizer, mcq_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

def generate_summary(llama_model, text):
    prompt = f"""[INST] <<SYS>>
You are a helpful AI assistant that creates clear and concise summaries.
<</SYS>>

Please provide a brief summary of the following video transcript in 2-3 sentences:

{text}

Focus on the main points and key takeaways.[/INST]"""
    
    try:
        optimize_memory()
        response = llama_model(
            prompt,
            max_tokens=150,
            temperature=0.7,
            top_p=0.95,
            stop=["</s>", "[/INST]"],
            echo=False
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None
    
@st.cache_resource
def load_models():
    try:
        # Load Whisper model
        whisper_model = whisper.load_model("base")
        
        # Load Llama model for summarization
        llama_model = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            n_batch=512,
            verbose=False
        )
        
        # Load LaMini model for MCQ generation
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        mcq_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            device_map='cpu',
            torch_dtype=torch.float32,
            local_files_only=True
        )
        
        return whisper_model, llama_model, tokenizer, mcq_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

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

def split_text(text, max_length=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1
        if current_length <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_mcq(text, tokenizer, model, num_questions=10):
    chunks = split_text(text)
    mcqs = []
    
    for chunk in chunks:
        if len(mcqs) >= num_questions:
            break
            
        prompt = f"Generate a multiple choice question with 4 options and indicate the correct answer from this text: {chunk}"
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50
                )
            
            mcq = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if mcq and len(mcq.strip()) > 0:
                formatted_mcq = format_mcq(mcq)
                if formatted_mcq:
                    mcqs.append(formatted_mcq)
                    
        except Exception as e:
            st.warning(f"Error generating MCQ: {str(e)}")
            continue
            
    return mcqs[:num_questions]

def format_mcq(mcq_text):
    try:
        mcq_text = mcq_text.strip()
        
        if not ('?' in mcq_text and ('a)' in mcq_text.lower() or 'a.' in mcq_text.lower())):
            return None
            
        mcq_text = mcq_text.replace('a)', '\na)')
        mcq_text = mcq_text.replace('A)', '\nA)')
        mcq_text = mcq_text.replace('a.', '\na.')
        mcq_text = mcq_text.replace('A.', '\nA.')
        
        return mcq_text
    except:
        return None

def save_to_pdf(title, content, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(10)
    
    # Add content
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    
    pdf.output(filename)

def main():
    st.title("🎥 YouTube Video Summary & MCQ Generator")
    
    # Load models
    with st.spinner("Loading AI models... This may take a minute..."):
        whisper_model, llama_model, tokenizer, mcq_model = load_models()
        st.success("✅ Models loaded successfully!")
    
    # YouTube URL input
    youtube_url = st.text_input("Enter YouTube URL", help="Paste the full YouTube video URL here")
    
    if st.button("Generate Summary & MCQs", type="primary"):
        if not youtube_url:
            st.warning("Please enter a YouTube URL")
            return
        
        try:
            # Download audio
            with st.spinner("📥 Downloading video audio..."):
                audio_path = download_audio(youtube_url)
                if not audio_path:
                    return
                st.success("✅ Audio downloaded successfully!")
            
            # Transcribe audio
            with st.spinner("🎯 Transcribing audio to text..."):
                transcript = transcribe_audio(whisper_model, audio_path)
                if not transcript:
                    return
                st.success("✅ Transcription complete!")
            
            # Generate summary
            with st.spinner("✍️ Generating summary..."):
                summary = generate_summary(llama_model, transcript)
                if not summary:
                    return
                
                st.success("✅ Summary generated!")
                st.subheader("📝 Summary")
                st.markdown(summary)
                
                # Save summary to PDF
                summary_pdf_path = "video_summary.pdf"
                save_to_pdf("YouTube Video Summary", summary, summary_pdf_path)
            
            # Generate MCQs
            with st.spinner("❓ Generating Multiple Choice Questions..."):
                mcqs = generate_mcq(transcript, tokenizer, mcq_model)
                
                if mcqs:
                    st.subheader("❓ Generated MCQs:")
                    mcq_text = "\n\n".join([f"Question {i+1}:\n{mcq}" for i, mcq in enumerate(mcqs)])
                    
                    # Save MCQs to PDF
                    mcq_pdf_path = "video_mcqs.pdf"
                    save_to_pdf("YouTube Video MCQs", mcq_text, mcq_pdf_path)
                    
                    # Display MCQs
                    for i, mcq in enumerate(mcqs, 1):
                        st.markdown(f"**Question {i}:**")
                        st.write(mcq)
                        st.write("---")
                    
                    # Provide download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(summary_pdf_path, "rb") as f:
                            st.download_button(
                                label="Download Summary PDF",
                                data=f,
                                file_name="video_summary.pdf",
                                mime="application/pdf"
                            )
                    
                    with col2:
                        with open(mcq_pdf_path, "rb") as f:
                            st.download_button(
                                label="Download MCQs PDF",
                                data=f,
                                file_name="video_mcqs.pdf",
                                mime="application/pdf"
                            )
                
                else:
                    st.error("Failed to generate MCQs.")
            
            # Cleanup temporary files
            try:
                os.remove(audio_path)
            except Exception as e:
                st.warning(f"Could not delete temporary audio file: {str(e)}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()