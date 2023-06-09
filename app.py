__all__ = ['learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'intf', 'multi_classifycation']

from fastai.vision.all import *
import gradio as gr
import timm

def get_y(path): return path.parent.name.split('_')

#load model
learn = load_learner('Model_ConvNext_Base.pkl')

categories = learn.dls.vocab

def tensor2labels(img):
    output = learn.predict(img)
    index = torch.where(output[1].sigmoid()>0.4)[0]
    category = categories[index]
    print(category)
    if 'dog' in category:
        category.remove('dog')
        if len(category)==0:
           return f"This is a Dog and I am not smart enough to know what breed it is :(("
        return f"This is a Dog and belongs to the breed: {category.pop()}"
    elif 'cat' in category:
        category.remove('cat')
        if len(category)==0:
            return f"This is a Cat and I am not smart enough to know what breed it is :(("
        return f"This is a Cat and belongs to the breed: {category.pop()}"
    else:
      raise Exception


gr.components

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['test.jpg', 'test1.jpg','test2.jpg','test3.jpg','test4.jpg','test5.jpg']

intf = gr.Interface(fn=tensor2labels, inputs=image, outputs=label, examples=examples)
intf.launch()