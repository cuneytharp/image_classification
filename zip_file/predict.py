# LOAD NEEDED LIBRARIES 
import argparse
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def process_image(image,image_size): 
    """
    This function returns image scaled for tensorflow
    Image is divided to 255 because color codes are between 0-255 and func returns values between 0-1
    """
    image = tf.cast(image, tf.float32)
    image= tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image
    

def predict(image_path:str, model:str, top_k=5):
    """
    This function makes prediction with provided model and returns the probabilities and classes as output
    """
    image = Image.open(image_path)
    image = np.asarray(image)
    image = np.expand_dims(image,  axis=0)
    image = process_image(image,image_size)
    probability_list = model.predict(image)
       
    classes = []
    probabilities = []
    
    rank = probability_list[0].argsort()[::-1]
    for i in range(top_k):
        index = rank[i] + 1
        class_name = name_of_classes[str(index)]
        probabilities.append(probability_list[0][index])
        classes.append(class_name)
    return probabilities, classes


if __name__ == '__main__':
    image_size = 224
    name_of_classes = {}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('modelname')
    parser.add_argument('--top_k')
    parser.add_argument('--cat_names') 
    #Loading category_class document
    
    arguments = parser.parse_args()
    for arg in arguments.__dict__:
        print("Provided argument",arguments.__dict__[arg])
    with open(arguments.cat_names, 'r') as f:
        name_of_classes = json.load(f)

    image_path = arguments.filename
    
    model = tf.keras.models.load_model(arguments.modelname ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = arguments.top_k
    if top_k is None: 
        top_k = 5

   
    probabilities, classes = predict(image_path, model, top_k)
    
    print("Summary:")
    for prob,clas in zip(probabilities,classes):
        print("Probability for",clas," is ", prob)