#!/usr/bin/env python3
"""
Gozumol Demo Script

This script demonstrates the basic usage of the Gozumol VisionAssistant
for analyzing images and providing navigation guidance.

Usage:
    # With GPU (default if available)
    python demo_assistant.py --image path/to/image.jpg

    # Force CPU mode
    python demo_assistant.py --image path/to/image.jpg --device cpu

    # Use a specific scenario
    python demo_assistant.py --image path/to/image.jpg --scenario traffic

    # Use an image URL
    python demo_assistant.py --image https://example.com/image.jpg
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gozumol Vision Assistant Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_assistant.py --image street.jpg
    python demo_assistant.py --image street.jpg --device cpu
    python demo_assistant.py --image street.jpg --scenario traffic
    python demo_assistant.py --image https://example.com/street.jpg
        """
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file or URL to analyze"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        choices=[
            "default", "outdoor", "indoor", "traffic",
            "crowd", "quick", "detailed", "safety", "crossing"
        ],
        default="default",
        help="Navigation scenario to use (default: default)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)"
    )

    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low memory mode for limited RAM/VRAM"
    )

    parser.add_argument(
        "--show-timing",
        action="store_true",
        help="Show detailed timing information"
    )

    return parser.parse_args()


def main():
    """Main entry point for the demo script."""
    args = parse_args()

    print("=" * 60)
    print("Gozumol Vision Assistant Demo")
    print("=" * 60)

    # Import here to show loading progress after args are parsed
    from gozumol import VisionAssistant
    from gozumol.core.utils import get_device_info

    # Show device information
    print("\nDevice Configuration:")
    device_info = get_device_info()
    print(f"  Device: {device_info['device'].upper()}")
    if device_info.get("cuda_device_name"):
        print(f"  GPU: {device_info['cuda_device_name']}")
        print(f"  VRAM: {device_info['cuda_memory_total_gb']:.1f} GB")

    # Initialize assistant
    print(f"\nInitializing assistant with scenario: {args.scenario}")
    assistant = VisionAssistant(
        device=args.device if args.device != "auto" else None,
        scenario=args.scenario,
        max_new_tokens=args.max_tokens,
        low_memory_mode=args.low_memory,
    )

    # Analyze image
    print(f"\nAnalyzing image: {args.image}")
    print("-" * 60)

    try:
        if args.show_timing:
            response, timing = assistant.describe(
                args.image,
                return_timing=True
            )

            print("\nAssistant Response:")
            print("-" * 40)
            print(response)
            print("-" * 40)

            print("\nTiming Information:")
            print(f"  Processor time: {timing['processor_time']:.3f}s")
            print(f"  Generation time: {timing['generation_time']:.3f}s")
            print(f"  Decode time: {timing['decode_time']:.3f}s")
            print(f"  Total time: {timing['total_time']:.3f}s")
        else:
            response = assistant.describe(args.image)

            print("\nAssistant Response:")
            print("-" * 40)
            print(response)
            print("-" * 40)

    except FileNotFoundError:
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing image: {e}")
        sys.exit(1)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
