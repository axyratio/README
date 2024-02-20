from diffusers import DiffusionPipeline as DP
from PIL import Image, ImageDraw, ImageFont
import torch

def text_to_image(text, diffuser_model):
    # image =Image.fromarray(image_data)
    # image.show()
    dp = DP.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

    image_data = dp(text.images[0])
    image = Image.fromarray(image_data)
    image.show()

if __name__ == "__main__":
    input_text = "Hello"
    diffuser_model = "example_diffuser_model"
    text_to_image(input_text, diffuser_model)
