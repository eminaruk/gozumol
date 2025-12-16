# Gozumol - AI-Powered Visual Assistance for the Visually Impaired

**Gozumol** (Turkish for "Be My Eyes") is an open-source library designed to help visually impaired individuals navigate their daily lives more safely and independently. Using a head-mounted camera and real-time AI analysis, Gozumol acts as a digital companion that describes the environment, warns about potential hazards, and provides actionable guidance.

## Vision

The goal of Gozumol is to create a reliable, real-time visual assistance system that:

- **Describes the environment** in a friendly, conversational tone
- **Warns about potential dangers** such as moving vehicles, obstacles, or hazardous conditions
- **Provides actionable guidance** like "wait", "step aside", or "be careful"
- **Avoids unnecessary details** to keep information concise and immediately useful
- **Speaks directly to the user** as a calm and trustworthy companion

## Why Phi-4 Multimodal?

After evaluating multiple Vision-Language Models (VLMs), we selected **Microsoft's Phi-4 Multimodal** as the primary model for this project. Here's why:

| Criteria | Phi-4 Multimodal | Qwen2-VL-2B |
|----------|------------------|-------------|
| **Safety Guidance** | Excellent - provides clear warnings and action-oriented instructions | Moderate - mostly descriptive |
| **Action-Oriented** | Yes - "wait", "be careful", "make room" | Limited |
| **Concise Output** | Good balance of context and brevity | Sometimes verbose |
| **Conversational Tone** | Warm, companion-like | More formal |
| **Direct User Address** | Consistently speaks to the user | Inconsistent |

Phi-4 Multimodal excels at providing the kind of immediate, safety-focused feedback that is critical for real-time assistance.

## Features

### Current Implementation

- **Image Analysis**: Process camera frames and generate environment descriptions
- **Safety Warnings**: Automatic detection and warning about potential hazards
- **Customizable Prompts**: Easily modify the AI's behavior through prompt engineering
- **CPU/GPU Support**: Run on various hardware configurations
- **Edge-Ready Architecture**: Designed for future deployment on edge devices

### Planned Features

- Real-time video stream processing
- Audio output integration (text-to-speech)
- Multi-language support
- Object tracking and movement prediction
- Indoor navigation assistance
- Integration with wearable devices

## Installation

```bash
# Clone the repository
git clone https://github.com/eminaruk/gozumol.git
cd gozumol

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.6.0+
- CUDA 12.x (for GPU acceleration, optional)
- 16GB+ RAM (32GB recommended for optimal performance)

## Quick Start

### Basic Usage

```python
from gozumol import VisionAssistant

# Initialize the assistant
assistant = VisionAssistant()

# Analyze an image
from PIL import Image
image = Image.open("path/to/image.jpg")

# Get environment description
description = assistant.describe(image)
print(description)
```

### Using Custom Prompts

```python
from gozumol import VisionAssistant
from gozumol.prompts import OUTDOOR_NAVIGATION_PROMPT

assistant = VisionAssistant(system_prompt=OUTDOOR_NAVIGATION_PROMPT)
description = assistant.describe(image)
```

### Running the Demo Script

```bash
# With GPU
python examples/demo_assistant.py --image path/to/image.jpg --device cuda

# With CPU only
python examples/demo_assistant.py --image path/to/image.jpg --device cpu
```

## Project Structure

```
gozumol/
├── gozumol/                 # Main library package
│   ├── __init__.py
│   ├── core/                # Core functionality
│   │   ├── __init__.py
│   │   ├── assistant.py     # Main VisionAssistant class
│   │   ├── model.py         # Model loading and inference
│   │   └── utils.py         # Utility functions
│   └── prompts/             # Prompt templates
│       ├── __init__.py
│       └── templates.py     # System and user prompts
├── examples/                # Example scripts
│   └── demo_assistant.py    # Demo application
├── notebooks/               # Jupyter notebooks for testing
│   ├── phi4_multimodal_test.ipynb
│   └── qwen2_vl_test.ipynb
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── README.md
```

## Prompts

The library includes carefully crafted prompts optimized for visual assistance:

- **Default Navigation Prompt**: General outdoor navigation assistance
- **Traffic Safety Prompt**: Enhanced focus on traffic and road safety
- **Indoor Navigation Prompt**: Optimized for indoor environments
- **Crowd Navigation Prompt**: For busy, crowded areas

See [`gozumol/prompts/templates.py`](gozumol/prompts/templates.py) for all available prompts.

## Contributing

We welcome contributions from the community! Here's how you can help:

1. **Report Issues**: Found a bug or have a suggestion? Open an issue.
2. **Improve Prompts**: Help us craft better prompts for specific scenarios.
3. **Add Features**: Submit pull requests with new functionality.
4. **Test & Feedback**: Test the system and provide feedback.
5. **Documentation**: Help improve our documentation.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black gozumol/
```

## Model Comparison Notes

During development, we tested multiple models. Here are sample outputs for the same street scene:

**Phi-4 Multimodal:**
> "Hey, you're on a cobblestone street in what looks like a lively neighborhood. There are pedestrians and parked bicycles around you, so be mindful of your step. Ahead, there's a cyclist approaching - give them some room. The traffic light is red, so let's wait here for now."

**Qwen2-VL-2B:**
> "Hello! It looks like you're in a lively city street at dusk. The buildings are beautifully lit up, and there are lots of people walking and riding bikes. The street is busy with traffic, including cars and a motorcycle."

Notice how Phi-4 provides more actionable guidance ("wait", "give them room") while Qwen focuses more on scene description.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Microsoft for the Phi-4 Multimodal model
- The Hugging Face team for the Transformers library
- All contributors and testers

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Gozumol** - Empowering independence through AI vision.
