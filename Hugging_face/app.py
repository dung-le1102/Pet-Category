from fastbook import *

from fastai.vision.all import *
import gradio as gr

#cannot instantiate 'WindowsPath' on your system
import pathlib
plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

#load model
learn =load_learner('convnext_model2.pkl') 



title = "Categories Pets"
description = """
<div style='text-align: center' ><p style='color:red ;font-size:20px' > The bot was trained to base on The Oxford-IIIT Pet Dataset which have 37 category pet dataset with roughly 200 images for each class
</p><a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">For more information</a></div>
"""
categories = learn.dls.vocab

def classify_img(img):
    
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))


image = gr.Image(shape = (192,192))
label = gr.Label()
examples = ['dog.jpg','bulldog.jpeg','chihuahua.jpg','sphynx.jpeg']

intf= gr.Interface(fn = classify_img, inputs = image, outputs =label,theme='JohnSmith9982/small_and_pretty',title=title,description=description ,examples = examples)
intf.launch(inline = False)
