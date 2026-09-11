"""VideoSeek core pipeline: download YouTube audio, transcribe it, and find the
segment that best matches a natural-language prompt.

UI-agnostic: used by both the Streamlit app (st.py) and the CLI (cli.py).
Errors are raised as exceptions; the callers decide how to display them.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import torch
import whisper
import yt_dlp
from sentence_transformers import SentenceTransformer

MAX_DURATION_SECONDS = 1800
SENTENCE_MODEL = "multi-qa-distilbert-cos-v1"


class VideoTooLong(Exception):
    def __init__(self, duration, limit=MAX_DURATION_SECONDS):
        super().__init__(f"Video is {duration}s long, limit is {limit}s")
        self.duration = duration
        self.limit = limit


class DownloadFailed(Exception):
    """All download attempts failed. `details` holds per-attempt errors plus diagnostics."""

    def __init__(self, errors, diagnostics):
        super().__init__("Failed to download audio")
        self.errors = errors
        self.diagnostics = diagnostics

    @property
    def details(self):
        return "\n\n".join(self.errors) + "\n\n" + self.diagnostics


def _deno_path():
    return shutil.which("deno") or os.path.join(sys.prefix, "bin", "deno")


def yt_dlp_extra_opts(cookies=None, proxy=None):
    """Options that make yt-dlp work from a cloud host.

    - JS runtime: yt-dlp needs one for full YouTube support. The `deno` pip
      package installs a binary into the venv's bin dir, so point at it directly.
    - cookies: contents of a Netscape-format cookies.txt, or a path to one.
      YouTube blocks datacenter IPs unless a logged-in session is provided.
    - proxy: proxy URL passed straight to yt-dlp.
    """
    opts = {}

    deno = _deno_path()
    if os.path.exists(deno):
        opts["js_runtimes"] = {"deno": {"path": deno}}

    if cookies:
        if os.path.exists(cookies):
            opts["cookiefile"] = cookies
        else:
            cookie_path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
            with open(cookie_path, "w") as f:
                f.write(cookies)
            opts["cookiefile"] = cookie_path

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


def diagnostics():
    deno = _deno_path()
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
# clients when the request comes from a datacenter IP, which shows up as
# "HTTP Error 403: Forbidden" on the media download. The web_embedded client
# does not require that token, so retry with it.
CLIENT_ATTEMPTS = [
    {},  # yt-dlp defaults
    {"extractor_args": {"youtube": {"player_client": ["web_embedded"]}}},
]


def download_youtube_audio(url, cookies=None, proxy=None, max_duration=MAX_DURATION_SECONDS):
    """Download the audio track of a YouTube video as m4a. Returns the file path.

    Raises VideoTooLong or DownloadFailed.
    """
    output_file = f"audio_{int(time.time())}.m4a"
    base_opts = {
        "format": "m4a/bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
        "outtmpl": output_file,
        "noplaylist": True,
        **yt_dlp_extra_opts(cookies=cookies, proxy=proxy),
    }

    errors = []
    for attempt in CLIENT_ATTEMPTS:
        logger = _CollectLogger()
        ydl_opts = {**base_opts, **attempt, "logger": logger}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                duration = info_dict["duration"]
                if duration > max_duration:
                    raise VideoTooLong(duration, max_duration)
                ydl.download([url])
            return output_file
        except VideoTooLong:
            raise
        except Exception as e:
            client = attempt.get("extractor_args", {}).get("youtube", {}).get(
                "player_client", ["default"]
            )
            msg = f"[client={','.join(client)}] {type(e).__name__}: {e}"
            if logger.lines:
                msg += "\n  " + "\n  ".join(logger.lines)
            errors.append(msg)
            print(f"[yt-dlp] {msg}", file=sys.stderr)
            for f in (output_file, output_file + ".part"):
                if os.path.exists(f):
                    os.remove(f)

    raise DownloadFailed(errors, diagnostics())


def transcribe_audio(audio_file, model="tiny"):
    """Transcribe with Whisper. Returns the list of segments (start, end, text)."""
    whisper_model = whisper.load_model(model)
    return whisper_model.transcribe(audio_file)["segments"]


def load_sentence_transformer_model():
    return SentenceTransformer(SENTENCE_MODEL)


def find_prompt_in_transcription(segments, prompt, model):
    """Similarity between the prompt and each pair of consecutive segments."""
    prompt_embedding = model.encode(prompt)
    texts = [segments[i]["text"] + segments[i + 1]["text"] for i in range(len(segments) - 1)]
    text_embeddings = model.encode(texts)
    return model.similarity(prompt_embedding, text_embeddings)


def best_match(segments, prompt, model, top_k=1):
    """Return the top_k matches as dicts with start, score and text, best first."""
    if len(segments) < 2:
        raise ValueError("Transcription too short to search")
    scores = find_prompt_in_transcription(segments, prompt, model).flatten()
    top = torch.topk(scores, min(top_k, scores.numel()))
    return [
        {
            "start": segments[i]["start"],
            "score": float(s),
            "text": (segments[i]["text"] + segments[i + 1]["text"]).strip(),
        }
        for s, i in zip(top.values.tolist(), top.indices.tolist())
    ]


def video_id_from_url(url):
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def seek(url, prompt, whisper_model="tiny", top_k=1, cookies=None, proxy=None, keep_audio=False):
    """Full pipeline. Returns (matches, segments)."""
    audio_file = download_youtube_audio(url, cookies=cookies, proxy=proxy)
    try:
        segments = transcribe_audio(audio_file, model=whisper_model)
    finally:
        if not keep_audio and os.path.exists(audio_file):
            os.remove(audio_file)
    model = load_sentence_transformer_model()
    return best_match(segments, prompt, model, top_k=top_k), segments
