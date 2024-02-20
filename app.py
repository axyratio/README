import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image
from diffusers import DiffusionPipeline as DP

bannerr = st.image("https://picsum.photos/800/500")
text = st.text_input("Prompt: ")
if text:
    dp = DP.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    image_data = dp(text).images[0]
    image = Image.fromarray(image_data)
    image.show()
    # st.image("https://picsum.photos/800/500")
    st.write(f"กำลังสร้างภาพ...")

# https://github.com/huggingface/diffusers
# https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube/data