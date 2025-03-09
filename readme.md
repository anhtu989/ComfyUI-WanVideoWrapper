markdown
# ComfyUI Wrapper Nodes for [WanVideo](https://github.com/Wan-Video/Wan2.1)

# WORK IN PROGRESS

## Installation

1. Clone this repository into the `custom_nodes` folder.
2. Install the required dependencies by running:
   bash
   pip install -r requirements.txt
   
   If you are using the portable installation, run the following command in the `ComfyUI_windows_portable` folder:
   bash
   python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt
   

## Models

Download the necessary models from the following Hugging Face repository:
- [WanVideo Models](https://huggingface.co/Kijai/WanVideo_comfy/tree/main)

Place the models in the appropriate directories:
- Text encoders: `ComfyUI/models/text_encoders`
- Transformer: `ComfyUI/models/diffusion_models`
- Vae: `ComfyUI/models/vae`

Alternatively, you can use the native ComfyUI text encoding and clip vision loader with the wrapper instead of the original models.

![image](https://github.com/user-attachments/assets/6a2fd9a5-8163-4c93-b362-92ef34dbd3a4)

---

## Examples

### TeaCache (Temporary WIP Naive Version, Waiting on the Official One) I2V:

[TeaCache Example](https://github.com/user-attachments/assets/504a9a50-3337-43d2-97b8-8e1661f29f46)

### Context Window Test:

1025 frames using a window size of 81 frames, with 16 overlap. With the 1.3B T2V model, this used under 5GB VRAM and took 10 minutes to generate on a 5090:

[Context Window Test Example](https://github.com/user-attachments/assets/89b393af-cf1b-49ae-aa29-23e57f65911e)

---

This very first test was 512x512x81:

~16GB used with 20/40 blocks offloaded:

[First Test Example](https://github.com/user-attachments/assets/fa6d0a4f-4a4d-4de5-84a4-877cc37b715f)

### Vid2Vid Example:

With 14B T2V model:

[14B T2V Model Example](https://github.com/user-attachments/assets/ef228b8a-a13a-4327-8a1b-1eb343cf00d8)

With 1.3B T2V model:

[1.3B T2V Model Example](https://github.com/user-attachments/assets/4f35ba84-da7a-4d5b-97ee-9641296f391e)