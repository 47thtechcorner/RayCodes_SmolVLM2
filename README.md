<div align="center">
  <a href="https://youtu.be/J4mJJW0ce_4">
    <img src="https://img.youtube.com/vi/J4mJJW0ce_4/0.jpg" alt="SmolVLM2 Dropped: 100% Offline Video AI on Laptop, iPhone & 4GB GPUs!">
  </a>
  <h3>📺 <a href="https://youtu.be/J4mJJW0ce_4">Watch the full tutorial on YouTube</a></h3>
</div>

# SmolVLM2: Compact On-Device Video Vision-Language Model

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model Family](https://img.shields.io/badge/Parameters-256M%20%7C%20500M%20%7C%202.2B-success.svg)](https://huggingface.co/blog/smolvlm2)
[![Video-MME](https://img.shields.io/badge/Video--MME-52.1-orange.svg)](https://huggingface.co/blog/smolvlm2)
[![VRAM Footprint](https://img.shields.io/badge/VRAM-1.38GB%20Peak%20(256M)-purple.svg)](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)
[![Edge Ready](https://img.shields.io/badge/Platform-CUDA%20%7C%20MLX%20%7C%20WebGPU-teal.svg)](https://huggingface.co/blog/smolvlm2)

> *"A compact family of open-source vision-language models developed by Hugging Face specifically for on-device video comprehension. By pairing a shape-optimized SigLIP vision encoder with a SmolLM2 language backbone, SmolVLM2 compresses 50 video frames into roughly 4,050 tokens, running smoothly on consumer laptops, 4GB GPUs, and smartphones while outperforming 2B-class competitors on Video-MME."*

---

## 📊 Production Test Workflows

| Workflow | Capability Tested | Input Artifact | Expected Deliverable |
| :--- | :--- | :--- | :--- |
| **Long-Form Video Chapter & Milestone Extractor** | Samples multi-frame temporal sequences to log topic transitions and key visual slide changes | Recorded technical architecture presentation video featuring voiceover walkthrough and live code demos | Interactive video chapter navigation visualizer with timestamped milestones and key topic summaries |
| **Multi-Obstacle Spatial Hazard Radar** | Evaluates spatial obstacle tracking while identifying small-obstacle blindspots | Real-world multi-obstacle street scene with moving vehicles, crossing pedestrians, and bicycles | Interactive spatial hazard radar dashboard with collision risk levels and blindspot logs |
| **Classroom Action & Activity Search** | Indexes multi-person activity logs using natural language queries for sub-second retrieval | Classroom session video featuring participants seated at desks, raising hands, and standing to speak | Interactive visual action search library with instant query matching and event cards |

---

## ⚡ Quick Start (Windows PowerShell)

### Step 1: Setup Environment & Download Model Weights

```powershell
# Create and activate clean Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA acceleration and multimodal dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.48.0" accelerate opencv-python pillow decord

# Download SmolVLM2-256M-Video-Instruct weights locally
hf download HuggingFaceTB/SmolVLM2-256M-Video-Instruct --local-dir models/SmolVLM2-256M-Video-Instruct
```

### Step 2: Run Production Pipeline

```powershell
python main.py
```

---

## 💡 Practical Use Cases

- 🛡️ **Air-Gapped Security Surveillance**: Process closed-circuit TV (CCTV) feeds entirely on local premise hardware with zero video streaming to cloud servers.
- 📱 **On-Device Smartphone Video Search**: Search personal phone video galleries using natural-language queries like *"biking in the rain"* directly on mobile neural engines.
- ⚡ **Automated Video Editing Assistant**: Index raw creator video archives and identify ideal B-roll cut points without paying expensive cloud API token fees.
- 🎓 **Lecture & Meeting Milestone Generator**: Automatically divide 1-hour university lectures or recorded corporate meetings into searchable topic chapters.
- 🚗 **Robotics & Drone Perception**: Run fast multi-frame visual verification on edge computing units for obstacle detection and contextual path navigation.

---

## 🔮 Future Feature Roadmap

- ⚡ **Dynamic Frame Rate Adaptation**: Intelligent frame decimation that samples keyframes dynamically based on optical flow scene velocity.
- 🎯 **Fine-Grained Spatial Bounding Coordinates**: Native bounding box regression for pinpoint object grounding in edge robotics applications.
- 🌐 **Browser-Native WebGPU Deployment**: Client-side video indexing running entirely in the browser via WebAssembly and WebGPU without backend servers.
- 🎙️ **Synchronized Audio-Visual Comprehension**: Joint audio spectrogram and visual frame processing for unified speech-and-vision event tagging.
- 📱 **Native Apple MLX Mobile Optimization**: Quantized 4-bit INT4 pipelines tailored specifically for iPhone and iPad local video playback assistants.

---

## 🏷️ Keywords & Tags

`SmolVLM2`, `SmolVLM2 video`, `video AI`, `local video AI`, `video understanding`, `offline video AI`, `Qwen2 VL`, `SmolVLM2 vs Qwen`, `video LLM`, `vision language model`, `VLM video`, `local VLM`, `open source VLM`, `video summarizer`, `video analysis`, `video captioning`, `video QA`, `video question answering`, `scene detection`, `spatial reasoning`, `video reasoning`, `edge AI`, `on device AI`, `SmolVLM2 GGUF`, `SmolVLM2 Ollama`, `Vision Language`, `Multimodal AI`, `Video Understanding`, `computer vision`, `Qwen3 VL`

