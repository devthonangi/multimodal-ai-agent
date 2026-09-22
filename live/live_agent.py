import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.multimodal_agent import MultimodalAgent
from utils.frames_utils import frame_to_pil
from utils.video_stream import VideoStream


def main():
    parser = argparse.ArgumentParser(description="Describe sampled webcam or video frames")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames analyzed per second")
    parser.add_argument("--question", default="What is happening in this scene?")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    agent = MultimodalAgent()
    for frame in VideoStream(src=source, fps=args.fps).frames():
        response = agent.process_image(frame_to_pil(frame), args.question)
        print(f"Live insight: {response}")


if __name__ == "__main__":
    main()
