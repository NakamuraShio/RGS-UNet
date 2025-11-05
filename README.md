RGS-UNet (ResNet18-Ghost Module-SIMAM-UNet)
================================
A model for **high-speed and accurate wire segmentation** on **edge devices** (e.g., Jetson Orin series).

Developed based on a [research paper](https://www.mdpi.com/1424-8220/25/11/3551) published on June 4, 2025

Assembly and training can be done in Google Colab: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_37t5hxmjYZHaqc0pdh4y5CIC9WNjk90?usp=sharing)


## Annotation Examples
<p align="center">
  <img src="assets/frame_1.jpg" width="45%">
  <img src="assets/mask_1.jpg" width="45%">
</p>
<p align="center">
  <img src="assets/frame_2.jpg" width="45%">
  <img src="assets/mask_2.jpg" width="45%">
</p>
<p align="center">
  <img src="assets/frame_3.jpg" width="45%">
  <img src="assets/mask_3.jpg" width="45%">
</p>
<p align="center">
  <img src="assets/frame_4.jpg" width="45%">
  <img src="assets/mask_4.jpg" width="45%">
  <br>
  <em>Left: original image, right: segmentation mask</em>
  <br>
  <em>Annotation was performed for three power cables only, excluding the lower telephone lines.</em>
</p>

Project Structure
------------------------------
- `train.py` / `infer.py` — training and inference scripts
- `nn_architecture.py` — network architecture
- `nn_utils.py` — metrics, losses, and utilities
- `data_generator.py` - data generator for the neural network
- `config.py` — configuration of paths and hyperparameters
- `requirements.txt` — dependencies
- `__init__.py` — package initializer and public API definition
- `README.md` — project description 
- `assests/` — sample results and visualizations

**Input:** RGB 736×1280 (default; can be adjusted via `DIMENSIONS` in `config.py`)  
**Output:** Single-channel probability map (semantic segmentation mask)

Supports **mixed precision** inference (float16 with critical layers preserved in float32), enabled via the `MIXED_PRECISION` flag in `config.py`.

## Architecture Highlights:
- **GhostModule-based layers** replace standard convolutions; a **modified ResNet18 backbone** and several other optimizations reduce redundant parameters to **57% of the baseline UNet**, providing a substantial performance gain.
- Incorporation of **SIMAM attention** and the **Mish activation function** yields accuracy improvements beyond the standard UNet baseline.
- A **composite loss function** combining **DICE** (for smoother convergence), **Focal Loss** (for precise wire boundary detection), and **Tversky Loss** (for handling mask discontinuities).

<p align="center">
  <img src="assets/architecture.jpg" width="90%">
  <br>
  <em>RGS-UNet Architecture</em>
</p>

<p align="center">
  <img src="assets/metrics.jpg" width="90%">
  <br>
  <em>Comparative metric curves: (a) F1-Score comparison (b) IoU comparison across competing models</em>
</p>

Environment Setup
------------------------------
```bash
# Clone the repository
git clone https://github.com/NakamuraShio/RGS-UNet.git
cd RGS-UNet

# Create a virtual environment
python -m venv venv

# Activate it (choose your OS)
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Running Inference
------------------------------
Supports three modes of operation:
- Single image processing
- Video processing
- Batch processing for folders containing multiple images or videos

**Using Python import:**
```python
# input_path — path to an image, video file, or folder
# save_path — output path for the processed image/video with the segmentation mask

from infer import run_segmentation
run_segmentation(input_path, save_path)
```

**Using the command line:**
```python
# Single image
python -m RGS-UNet.infer --input "./materials/frame.jpg" --output "./results/"
# Single video file
python -m RGS-UNet.infer --input "./materials/video.mp4" --output "./results/"
# Batch processing of all files in a folder
python -m RGS-UNet.infer --input "./materials" --output "./results/"
```

Model Training
------------------------------
```python
# Example: via import
from train import train_model
history = train_model(dataset_path, weights_path)

# Example: via terminal
python -m RGS-UNet.train --dataset "./dataset" --weights "./RGS-UNet/weights"
```


Author
------------------------------
Project by: **Artem Almazov**  
Computer Vision Developer

Email: aaalmaz@gmail.com  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/artem-almazov/)  
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/NakamuraShio)