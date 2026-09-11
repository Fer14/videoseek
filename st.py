import streamlit as st
from PIL import Image
import re
import whisper
import yt_dlp
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction
import time
import os
import sys
import shutil
import tempfile
import subprocess


def _yt_dlp_extra_opts():
    """Options that make yt-dlp work from a cloud host (Streamlit Community Cloud).

    - JS runtime: yt-dlp needs one for full YouTube support. The `deno` pip
      package installs a binary into the venv's bin dir, so point at it directly.
    - Cookies: YouTube blocks datacenter IPs with "Sign in to confirm you're not
      a bot". Store a Netscape-format cookies.txt in st.secrets["YOUTUBE_COOKIES"]
      to get past that. Optionally set st.secrets["YOUTUBE_PROXY"].
    """
    opts = {}

    deno = shutil.which("deno") or os.path.join(sys.prefix, "bin", "deno")
    if os.path.exists(deno):
        opts["js_runtimes"] = {"deno": {"path": deno}}

    cookies = st.secrets.get("YOUTUBE_COOKIES") if hasattr(st, "secrets") else None
    if cookies:
        cookie_path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
        with open(cookie_path, "w") as f:
            f.write(cookies)
        opts["cookiefile"] = cookie_path

    proxy = st.secrets.get("YOUTUBE_PROXY") if hasattr(st, "secrets") else None
    if proxy:
        opts["proxy"] = proxy

    return opts


class _CollectLogger:
    """yt-dlp logger that keeps warnings/errors so we can show them to the user."""

    def __init__(self):
        self.lines = []

    def debug(self, msg):
        if msg.startswith("[debug] ") or "[jsc" in msg or "[pot" in msg:
            self.lines.append(msg)

    def info(self, msg):
        pass

    def warning(self, msg):
        self.lines.append(f"WARNING: {msg}")
        print(f"[yt-dlp] WARNING: {msg}", file=sys.stderr)

    def error(self, msg):
        self.lines.append(f"ERROR: {msg}")
        print(f"[yt-dlp] ERROR: {msg}", file=sys.stderr)


def _diagnostics():
    deno = shutil.which("deno") or os.path.join(sys.prefix, "bin", "deno")
    try:
        deno_ver = subprocess.run(
            [deno, "--version"], capture_output=True, text=True, timeout=20
        ).stdout.splitlines()[0]
    except Exception as e:
        deno_ver = f"deno failed to run: {type(e).__name__}: {e}"
    return (
        f"yt-dlp {yt_dlp.version.__version__}, python {sys.version.split()[0]}, "
        f"{deno_ver} at {deno}"
    )


# Attempt order. YouTube requires a "proof of origin" token for the default web
# clients when the request comes from a datacenter IP (Streamlit Cloud), which
# shows up as "HTTP Error 403: Forbidden" on the media download. The
# web_embedded client does not require that token, so retry with it.
_CLIENT_ATTEMPTS = [
    {},  # yt-dlp defaults
    {"extractor_args": {"youtube": {"player_client": ["web_embedded"]}}},
]


def download_youtube_audio(url, output_file="audio2.m4a"):
    timestamp = int(time.time())  # Unix timestamp
    output_file = f"audio_{timestamp}.m4a"
    # Define yt_dlp options
    base_opts = {
        "format": "m4a/bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "outtmpl": output_file,  # Save output file as specified
        "noplaylist": True,
        **_yt_dlp_extra_opts(),
    }

    errors = []
    for attempt in _CLIENT_ATTEMPTS:
        logger = _CollectLogger()
        ydl_opts = {**base_opts, **attempt, "logger": logger}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                duration_seconds = info_dict["duration"]
                if duration_seconds > 1800:
                    st.error(
                        "Are you trying to break my website? 🤨 Video is too long! Update your plan or send me a bizum "
                    )
                    st.stop()
                ydl.download([url])
            return output_file
        except Exception as e:
            client = attempt.get("extractor_args", {}).get("youtube", {}).get(
                "player_client", ["default"]
            )
            msg = f"[client={','.join(client)}] {type(e).__name__}: {e}"
            if logger.lines:
                msg += "\n  " + "\n  ".join(logger.lines)
            errors.append(msg)
            # Log the real error so it shows up in the Streamlit Cloud logs
            print(f"[yt-dlp] {msg}", file=sys.stderr)
            for f in (output_file, output_file + ".part"):
                if os.path.exists(f):
                    os.remove(f)

    st.error("Failed to download audio. Please check the URL and try again.")
    with st.expander("Error details"):
        st.code("\n\n".join(errors) + "\n\n" + _diagnostics())
    st.stop()  # Stop further execution if download fails


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
