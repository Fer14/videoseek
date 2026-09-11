import streamlit as st
from PIL import Image
import os

import core


def secret(name):
    """st.secrets raises if no secrets.toml exists (local runs); treat that as unset."""
    try:
        return st.secrets.get(name)
    except Exception:
        return None


st.set_page_config(page_title="VideoSeek", page_icon="🌐")


logo = Image.open("logo.png")
st.image(logo, use_container_width=True)

# Display the app title and description
st.title("URL and Prompt Input App")
st.write("Enter a URL and a what do you want to seek in it")

st.warning(
    """
**Heads up: this app may not be able to download YouTube videos anymore.**

Since mid 2024 YouTube blocks download requests coming from cloud providers
(it answers *"Sign in to confirm you're not a bot"* and later returns
*HTTP 403* on the audio download). Home connections are mostly spared, which is
why VideoSeek works when run locally but usually fails on this Streamlit Cloud
deployment. Many Streamlit and Hugging Face apps built on
[yt-dlp](https://github.com/yt-dlp/yt-dlp) stopped working at that point.

To use VideoSeek reliably, clone the
[repository](https://github.com/Fer14/videoseek) and run it on your own machine.
    """,
    icon="⚠️",
)

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
        try:
            audio_file = core.download_youtube_audio(
                url_input,
                cookies=secret("YOUTUBE_COOKIES"),
                proxy=secret("YOUTUBE_PROXY"),
            )
        except core.VideoTooLong:
            st.error(
                "Are you trying to break my website? 🤨 Video is too long! Update your plan or send me a bizum "
            )
            st.stop()
        except core.DownloadFailed as e:
            st.error("Failed to download audio. Please check the URL and try again.")
            with st.expander("Error details"):
                st.code(e.details)
            st.stop()

        progress_bar.progress(60, "Transcribing audio...")
        # st.info("Transcribing audio...")
        try:
            segments = core.transcribe_audio(audio_file)
        except Exception as e:
            print(e)
            st.error("Failed to transcribe audio.")
            st.stop()

        progress_bar.progress(60, "Matching text..")
        # st.success("Audio downloaded and transcribed successfully!")
        try:
            model = core.load_sentence_transformer_model()
        except Exception:
            st.error("Failed to load sentence transformer model.")
            st.stop()
        start_time = core.best_match(segments, prompt_input, model)[0]["start"]
        progress_bar.progress(90, "Setting up the video")

        os.remove(audio_file)

        video_id = core.video_id_from_url(url_input)
        if video_id:
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
