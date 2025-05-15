# Models

## Vision Backbone (ViT)

This is a very lightweight Vision Transformer in native pytorch. I took inspiration from the following sources:
- https://github.com/karpathy/nanoGPT (General Transformer Decoder)
- https://arxiv.org/abs/2010.11929 (ViT Paper)
- https://arxiv.org/abs/2303.15343 (SigLiP Paper)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py (HF SigLiP Implementation)

## Language Model (Llama / SmolLM)

This is a decoder only LM, following the Llama 2/3 architecture. Inspiration from the following sources:
- https://arxiv.org/pdf/2307.09288 (Original Llama Paper)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py (HF Llama Implementation)

## Modality Projection

This is a simple MLP (Linear Layer) for the Modality Projection between the Image Patch Encodings and the Language Embedding Space with a simple Pixel Shuffle (https://arxiv.org/pdf/2504.05299)

## Vision-Language-Model

This brings all the individual parts together and handles the concatenation of images and text. Built as a simple version of SmolVLM (https://arxiv.org/pdf/2504.05299)
