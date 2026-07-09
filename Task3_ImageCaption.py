from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

img = Image.open("test.jpg").convert("RGB")
pixel_values = processor(images=img, return_tensors="pt").pixel_values
output = model.generate(pixel_values)
caption = tokenizer.decode(output[0], skip_special_tokens=True)
print("Caption:", caption)
