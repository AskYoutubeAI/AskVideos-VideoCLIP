

<p align="center">
  <img src="https://raw.githubusercontent.com/AskYoutubeAI/AskVideos-VideoCLIP/main/AskVideosVideoClipLogo.png" height="250" alt="AskVideos-VideoCLIP" />
</p>
<p align="center">
  <em>Joint Video-Text embeddings for search, classification and more.</em>
</p>
<p align="center">
    <a href='https://huggingface.co/AskYoutube/AskVideos-VideoCLIP-v0.1'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a>
    <a target="_blank" href="https://colab.research.google.com/drive/1kVzoQUS3phupujY-8Bym0nHezRRyd0YQ">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</p>
<p align="center">


  
# AskVideos-VideoCLIP
- AskVideos-VideoCLIP is a language-grounded video embedding model.
- This model produces a single context-aware embedding for each video clip.
- 16 frames are sampled from each video clip to generate a video embedding.
- The model is trained with contrastive and captioning loss to ground the video embeddings to text.
## Pre-trained & Fine-tuned Checkpoints
| Checkpoint       | Link |
|:------------|-------------|
| AskVideos-VideoCLIP-v0.1    | [link](https://huggingface.co/AskYoutube/AskVideos-VideoCLIP-v0.1) |
| AskVideos-VideoCLIP-v0.2    | [link](https://huggingface.co/AskYoutube/AskVideos-VideoCLIP-v0.2) |

The demo is also available to run on colab.
| Model       | Colab link |
|:------------|-------------|
| AskVideos-VideoCLIP-v0.1    | [link](https://colab.research.google.com/drive/1kVzoQUS3phupujY-8Bym0nHezRRyd0YQ) |
| AskVideos-VideoCLIP-v0.2    | [link](https://colab.research.google.com/drive/1TfEIqzEq_ppVSQHfEHXvbIrh0MTn9vpX?usp=sharing) |

## Usage
#### Environment Preparation
First, install ffmpeg.
```
apt update
apt install ffmpeg
```
Then, create a conda environment:
```
conda create -n askvideosclip python=3.9 
conda activate askvideosclip
```
Then, install the requiremnts:
```
pip3 install -U pip
pip3 install -r requirements.txt
```

## How to Run Demo Locally
```
python video_clip.py
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AskYoutubeAI/AskVideos-VideoCLIP&type=Date)](https://star-history.com/#AskYoutubeAI/AskVideos-VideoCLIP&Date)


## Term of Use
AskVideos code and models are distributed under the Apache 2.0 license.

## Acknowledgement
This model is inspired by the [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) Video-Qformer model.

## Citation
```
bibtex
@misc{askvideos2024videoclip,
  title        = {AskVideos-VideoCLIP: Language-grounded video embeddings},
  author       = {AskVideos},
  year         = {2024},
  howpublished = {GitHub},
  url          = {https://github.com/AskYoutubeAI/AskVideos-VideoCLIP}
}
