# M2Transformer-Full
M2Transformer including Faster R-CNN with model pretrained on Visual Genome

I use M2 Transformer code from  [meshed-memory-transformer](https://github.com/aimagelab/meshed-memory-transformer). However, the input of the code is features with `(N, 2048)`. The paper use `Faster R-CNN with model pretrained on Visual` as the feature extractor so I connect two of them. The feature extractor I use is [Faster-R-CNN-with-model-pretrained-on-Visual-Genome](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome).

The input will be an image and the output will be caption of the image.

Link models:

https://drive.google.com/drive/folders/17mqqZWNYyp40jvFoUsyn5ABo4mw_Dgnp?usp=sharing

Put all the downloaded models into ./models directory.

# Installation

		pip install -r requirement.txt

# Inference

		python gui.py
