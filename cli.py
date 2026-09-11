#!/usr/bin/env python3
"""VideoSeek command line: find where a YouTube video talks about something.

    python cli.py "https://www.youtube.com/watch?v=s9w3gtgvNSU" "embeddings"
"""

import argparse
import sys
import time

import core


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("prompt", help="What to look for, in natural language")
    parser.add_argument("--model", default="tiny", help="Whisper model size (default: tiny)")
    parser.add_argument("--top", type=int, default=3, help="Number of matches to show (default: 3)")
    parser.add_argument("--cookies", help="Path to a Netscape cookies.txt (needed on cloud/datacenter IPs)")
    parser.add_argument("--proxy", help="Proxy URL for yt-dlp")
    parser.add_argument("--keep-audio", action="store_true", help="Do not delete the downloaded audio file")
    args = parser.parse_args(argv)

    t0 = time.time()
    print("Downloading audio...", file=sys.stderr)
    try:
        matches, segments = core.seek(
            args.url, args.prompt, whisper_model=args.model, top_k=args.top,
            cookies=args.cookies, proxy=args.proxy, keep_audio=args.keep_audio,
        )
    except core.VideoTooLong as e:
        print(f"Video is too long ({fmt_time(e.duration)}); limit is {fmt_time(e.limit)}.", file=sys.stderr)
        return 2
    except core.DownloadFailed as e:
        print("Failed to download audio.\n", file=sys.stderr)
        print(e.details, file=sys.stderr)
        return 1

    video_id = core.video_id_from_url(args.url)
    best = matches[0]
    print(f"\nBest match at {fmt_time(best['start'])}  (score {best['score']:.3f})")
    print(f"  \"{best['text']}\"")
    if video_id:
        print(f"  https://www.youtube.com/watch?v={video_id}&t={int(best['start'])}s")
    if len(matches) > 1:
        print("\nOther candidates:")
        for m in matches[1:]:
            print(f"  {fmt_time(m['start']):>6}  {m['score']:.3f}  {m['text'][:100]}")
    print(f"\n{len(segments)} segments transcribed, {time.time() - t0:.0f}s total", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
