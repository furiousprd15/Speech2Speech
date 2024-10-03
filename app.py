import streamlit as st
import torch
from transformers import pipeline
# from transformers.utils import is_flash_attn_2_available
from openai import OpenAI  # Ensure this is the correct import for your OpenAI client
import streamlit.components.v1 as components  # Import the components module
import re
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
from new import context
import base64
from time import sleep
from TTS.api import TTS
import wave

# Configuration
st.set_page_config(page_title="Krutrim CCAI Platform", layout="wide")
vllm_url = "http://10.230.17.15:1591/v1/"
model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Initialize OpenAI client with custom base_url
client = OpenAI(api_key="EMPTY", base_url=vllm_url)

# Initialize Whisper ASR pipeline
@st.cache_resource
def load_model():
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device="cuda:0",
        # model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )
    return pipe

pipe = load_model()

@st.cache_resource
def load_tts():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return tts

tts = load_tts()

#Initialize Session State for Response Text and Buffer
if 'response_text' not in st.session_state:
    st.session_state['response_text'] = ""
if 'buffer' not in st.session_state:
    st.session_state['buffer'] = ""
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    
# Regular expression to detect sentence boundaries
sentence_end_re = re.compile(r'([.!?])\s')
st.markdown('<h1 class="title">🔊 Krutrim CCAI Onboarding', unsafe_allow_html=True)

# Audio Recorder
st.markdown('<div class="recorder-container">', unsafe_allow_html=True)
audio_bytes = audio_recorder(
    text="",
    icon_name="microphone",
    icon_size="6x"
)
st.markdown('</div>', unsafe_allow_html=True)

# Transcription Placeholder
transcription_placeholder = st.empty()

if audio_bytes:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    outputs = pipe(
        "temp_audio.wav",
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
        generate_kwargs={"language": "english"}
    )

    transcription = " ".join([segment['text'] for segment in outputs['chunks']])
    transcription_placeholder.write(f"**Transcription:** {transcription}")

    st.session_state['chat_history'].append({"role": "user", "content": transcription})
    user_question = transcription
else:
    user_question = None

# System Prompt
system_prompt = f"""You are a helpful and highly quirky assistant. You are provided with the following contextual information. 

Contextual information: 

{context}

Your task is to assist the user queries based on the provided information. For queries outside the domain of the contextual information, answer the query but then redirect the user to ask more about OLA services.

Keep the length of responses short in 30-40 words. Please stick to the role and context provided. Avoid providing quotes in your responses."""

# Function to call TTS and return the audio duration

import wave
from pydub import AudioSegment

# audio_file = AudioSegment.from_file("sample.wav")
# slow_sound = speed_change_pydub(audio_file, 0.75)
# fast_sound = speed_change_pydub(audio_file, 2.0)

def call_xtts(text, file_name="demo.wav"):
    tts.tts_to_file(text=text, file_path=file_name, speaker="Suad Qasim", language="en", speed = 5)
    # speed_change_wave(file_name,3)

    with wave.open(file_name, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)  # Duration in seconds
    with open(file_name, "rb") as audio_file:
        audio_data = audio_file.read()
    return audio_data, duration  # Return audio data and its duration

# Function to create audio HTML
def create_audio_html(audio_content):
    audio_base64 = base64.b64encode(audio_content).decode("utf-8")
    return f"""
        <audio id="audioPlayer" controls autoplay>
            <source src="data:audio/mpeg;base64,{audio_base64}">
            Your browser does not support the audio tag.
        </audio>
    """

# Process user question (if available) as input
if user_question:
    st.session_state['response_text'] = ""
    st.session_state['buffer'] = ""
    
    response_placeholder = st.empty()
    response_placeholder.markdown("**Response:**")
    
    messages = [{"role": "system", "content": system_prompt}] + [{"role": "user", "content": q["content"]} for q in st.session_state['chat_history']]

    # Function to stream vLLM response
    def get_vllm_stream(messages):
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=16000,
            top_p=0.9,
            stream=True
        )

    try:
        stream = get_vllm_stream(messages)
        response_text = ""
        first_sentence_played = False  # Flag to track if the first sentence is already played
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunk_text = chunk.choices[0].delta.content
                if chunk_text:
                    st.session_state['buffer'] += chunk_text
                    response_text += chunk_text
                    
                    # Check for sentence boundary
                    sentences = []
                    while True:
                        match = sentence_end_re.search(st.session_state['buffer'])
                        if match:
                            end_idx = match.end()
                            sentence = st.session_state['buffer'][:end_idx].strip()
                            sentences.append(sentence)
                            st.session_state['buffer'] = st.session_state['buffer'][end_idx:].strip()
                        else:
                            break
                    
                    if sentences:
                        for sentence in sentences:
                            st.session_state['response_text'] += sentence + " "
                            response_placeholder.markdown(f"**Response:** {st.session_state['response_text']}")
                            
                            if not first_sentence_played:
                                # Play first sentence and wait for its duration
                                audio_data, duration = call_xtts(sentence)
                                audio_html = create_audio_html(audio_data)
                                components.html(audio_html, height=0, width=0)
                                sleep(duration)  # Wait for the audio to complete
                                first_sentence_played = True
                            else:
                                # Play subsequent sentences, one at a time, waiting for each audio to complete
                                audio_data, duration = call_xtts(sentence)
                                audio_html = create_audio_html(audio_data)
                                components.html(audio_html, height=0, width=0)
                                sleep(duration)

        # After processing all chunks, check if there's anything left in the buffer
        if st.session_state['buffer']:
            remaining_text = st.session_state['buffer']
            st.session_state['response_text'] += remaining_text + " "
            response_placeholder.markdown(f"**Response:** {st.session_state['response_text']}")
            
            # Play the remaining text
            audio_data, duration = call_xtts(remaining_text)
            audio_html = create_audio_html(audio_data)
            components.html(audio_html, height=0, width=0)
            sleep(duration)  # Wait for the last audio to finish

        st.session_state['chat_history'].append({"role": "assistant", "content": response_text})

    except Exception as e:
        st.error(f"An error occurred: {e}")




