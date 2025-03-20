import torch
from transformers import pipeline
import whisper
import gradio as gr
from gtts import gTTS
from PIL import Image
import nltk
import re
import tempfile
import os
import multiprocessing

# Enable multiprocessing for MacOS
multiprocessing.freeze_support()

# Download NLTK data
nltk.download('punkt', quiet=True)

# Configure device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize a smaller vision model
model_id = "microsoft/git-base"  # Using a more stable model
print("Loading image captioning model...")
pipe = None  # We'll initialize this later to avoid multiprocessing issues

# Initialize Whisper model
print("Loading Whisper model...")
audio_model = None  # We'll initialize this later

def initialize_models():
    """Initialize models safely"""
    global pipe, audio_model
    if pipe is None:
        pipe = pipeline("image-to-text", model=model_id)
    if audio_model is None:
        audio_model = whisper.load_model("medium", device=DEVICE)
    return pipe, audio_model

def img2txt(input_text, input_image):
    """Process image with the vision model"""
    global pipe
    if pipe is None:
        pipe, _ = initialize_models()
    
    try:
        # Generate basic caption
        outputs = pipe(input_image)
        caption = outputs[0]['generated_text']
        
        # If there's a specific question, append it to the response
        if input_text and input_text.strip():
            response = f"Based on the image which shows {caption}, "
            response += f"addressing your question: {input_text}\n"
            return response
            
        return caption
    except Exception as e:
        print(f"Error in image processing: {str(e)}")
        return "Sorry, I couldn't process the image properly."

def transcribe(audio):
    """Transcribe audio using Whisper"""
    global audio_model
    if audio_model is None:
        _, audio_model = initialize_models()
    
    if audio is None:
        return ''
    
    try:
        audio = whisper.load_audio(audio)
        audio = whisper.pad_or_trim(audio)
        
        mel = whisper.log_mel_spectrogram(audio).to(audio_model.device)
        result = whisper.decode(audio_model, mel, whisper.DecodingOptions())
        
        return result.text
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return "Sorry, I couldn't transcribe the audio properly."

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return None

def process_inputs(audio, image):
    """Main processing function"""
    try:
        # Process speech to text
        speech_to_text_output = transcribe(audio) if audio is not None else ""
        
        # Process image and generate response
        if image is not None:
            query = speech_to_text_output if speech_to_text_output else "Describe this image in detail"
            chatgpt_output = img2txt(query, image)
        else:
            chatgpt_output = "No image provided."
        
        # Generate audio response
        audio_output = text_to_speech(chatgpt_output)
        
        return speech_to_text_output, chatgpt_output, audio_output
        
    except Exception as e:
        print(f"Error in process_inputs: {str(e)}")
        return str(e), str(e), None

# Create Gradio interface
demo = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Voice Input"),  # Fixed Audio component syntax
        gr.Image(type="pil", label="Image Input")  # Specified image type
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Image Analysis"),
        gr.Audio(label="AI Response", type="filepath")
    ],
    title="Image Analysis with Voice Interface",
    description="Upload an image and ask questions using your voice. The AI will analyze the image and respond with both text and speech."
)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    # Initialize models before launching the interface
    initialize_models()
    # Launch with minimal GPU memory usage
    demo.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
    )
