import streamlit as st
from PIL import Image
import re
import whisper
import yt_dlp
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction
import time
import os


def download_youtube_audio(url, output_file="audio2.m4a"):
    timestamp = int(time.time())  # Unix timestamp
    output_file = f"audio_{timestamp}.m4a"
    # Define yt_dlp options
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "outtmpl": output_file,  # Save output file as specified
    }

    # Download and extract audio
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            duration_seconds = info_dict["duration"]
            if duration_seconds <= 1800:
                ydl.download([url])
            else:
                st.error(
                    "Are you trying to break my website? 🤨 Video is too long! Update your plan or send me a bizum "
                )
    except Exception as e:
        st.error("Failed to download audio. Please check the URL and try again.")
        st.stop()  # Stop further execution if download fails

    return output_file


def transcribe_audio(audio_file, model="tiny"):
    try:
        model = whisper.load_model(model)
        # Transcribe audio
        transcription = model.transcribe(audio_file)
        return transcription["segments"]

    except Exception as e:
        st.error("Failed to transcribe audio.")
        st.stop()
        print(e)
        progress_bar.progress(0)


def find_prompt_in_transcription(segments, prompt, model):
    prompt_embedding = model.encode(prompt)
    text_embeddings = []
    for i in range(len(segments[:-1])):
        text = segments[i]["text"] + segments[i + 1]["text"]
        text_embedding = model.encode(text)
        text_embeddings.append(text_embedding)

    return model.similarity(prompt_embedding, text_embeddings)


def load_sentence_transformer_model():
    try:
        return SentenceTransformer("multi-qa-distilbert-cos-v1")
    except Exception as e:
        st.error("Failed to load sentence transformer model.")
        st.stop()


st.set_page_config(page_title="VideoSeek", page_icon="🌐")


logo = Image.open("logo.png")
st.image(logo, use_container_width=True)

# Display the app title and description
st.title("URL and Prompt Input App")
st.write("Enter a URL and a what do you want to seek in it")

# Create input fields
url_input = st.text_input(
    "Enter URL", placeholder="https://www.youtube.com/watch?v=zYQP1v8etDU"
)
prompt_input = st.text_area(
    "When does the video talks about...?", placeholder="da las gracias por verles"
)


# Submit button
if st.button("Submit"):
    if url_input and prompt_input:
        # Perform any action with the inputs
        st.write("URL entered:", url_input)
        st.write("Prompt entered:", prompt_input)

        # Example: Display a response message
        st.success("Your input has been received!")

        progress_bar = st.progress(0)
        progress_bar.progress(30, "Downloading audio...")
        # st.info("Downloading audio...")
        audio_file = download_youtube_audio(url_input)

        progress_bar.progress(60, "Transcribing audio...")
        # st.info("Transcribing audio...")
        segments = transcribe_audio(audio_file)

        progress_bar.progress(60, "Matching text..")
        # st.success("Audio downloaded and transcribed successfully!")
        model = load_sentence_transformer_model()
        result = find_prompt_in_transcription(
            segments, prompt_input, model
        )
        start_time = segments[torch.argmax(result).item()]["start"]
        progress_bar.progress(90, "Setting up the video")

        os.remove(audio_file)

        youtube_url_match = re.search(r"v=([A-Za-z0-9_-]+)", url_input)
        if youtube_url_match:
            video_id = youtube_url_match.group(1)
            print(f"https://www.youtube.com/embed/{video_id}?start={int(start_time)}")
            youtube_embed_html = f"""
<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%;">
    <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
            src="https://www.youtube.com/embed/{video_id}?start={int(start_time)}" 
            title="YouTube video player" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
    </iframe>
</div>
            """
            st.components.v1.html(youtube_embed_html, height=700, width=700)
            progress_bar.progress(
                100,
                "Success!",
            )
