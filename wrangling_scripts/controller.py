from wrangling_scripts.helper import *
from wrangling_scripts.load_model import *
import base64
import pathlib
import io
from PIL import Image


model, graphobject = init_model()
image_path = "C://work_local//Python//DataScientistND_Project6_Capstone_Project//Dog_Breed//sample_imgs//Bernhardiner.jpg"

with open(image_path, "rb") as image:
  f = image.read()
  b = bytearray(f)

image = Image.open(b)

predictions = preprocess_resnet(image)
results = decode_prob(predictions, top_probs=3)
print(results)