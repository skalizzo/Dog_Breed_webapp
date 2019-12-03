# Dog Breed Classification Webapp

This webapp uses the Flask framework to deploy a trained model based on ResNet50 
that is used to classify dog breeds in images. If you'd upload images of humans 
to the app it should recognise them as humans and give you the dog breed that 
resembles these humans most. 
 
This webapp is part of the Capstone project for the Udacity Data 
Scientist Nanodegree program. 

The corresponding Jupyter notebook where the model has been built can 
be found here: [https://github.com/skalizzo/Dog_Breed_Classification](https://github.com/skalizzo/Dog_Breed_Classification)

### Technologies used
<li>Python 3.7</li>
<li>Flask 1.1.1</li>
<li>Jinja2 2.10.3</li>
<li>Werkzeug 0.16.0</li>
<li>Keras 2.2.5</li>
<li>tensorflow 2.0.0 </li>
<li>OpenCV 4.1.2.30</li>
<li>numpy 1.16.1</li>

### Installation/Usage
1. Clone this repository
2. Install the required packages (see above)
3. Run with Python: $ python app.py
4. Open browser at http://localhost:5000, select an image on your computer 
and start the prediction using the "Predict Dog Breed" button

If you want to modify the UI you can modify the files in the templates directory.

### Project Definition
This webapp is part of the Capstone project for the Udacity Data 
Scientist Nanodegree program. The goal of this project is to classify images of dogs according to their breed. 
If a human is detected the code should provide an estimate of the dog breed that is most resembling. 

Success will be measured by the accuracy of predicting the correct dog breed for the given image.

### Analysis
Udacity has provided 8351 labeled dog images and 13233 images of humans to train the algorithm. There are 133 possible 
dog breeds that can be predicted. The classes in the training dataset are a little unbalanced. The dog breed with the 
lowest image count has 26 images while the dog breed with the highest image count has 77 images.

Here you can see the distribution of the train set:

![No of images per dog breed within the train set](readme_images/dog_breed_train.png)


### Methodology
The creation of the algorithm follows the following steps:
* Step 0: Import Datasets <br>
The 3 dog data sets for training, testing and validation are imported as numpy arrays.


* Step 1: Build a face detector for humans<br>
After I've imported the dataset I've implemented a face detecting algorithm based on OpenCV's 
implementation of Haar-based classifiers. It returns True if a human face is detected within the image.
Within a test using 100 sample human images and 100 sample dog images the face detector classified
 100% of the human images correctly which is great but it also classifies 11% of the dog images 
 wrongly as humans. That is not a great performance but it is ok.

* Step 2: Build a detector for dogs<br>
I then implemented an alghorithm to determine wether the image has a dog in it or not. To do that the image 
has to be preprocessed first (rescaling it and transforming it into a 4D tensor). Then the Resnet50 model 
(trained on the ImageNet database) is used to do the prediction. This algorithm works perfectly well as it 
classified 100% of the dog images as dogs and 0% of the humans as dogs.

* Step 3: Create a CNN to Classify Dog Breeds (from Scratch) <br>
First i've tried to create a convolutional neural network from scratch using the keras library. 

I've used a combination of Convolutional Layers and (local) Pooling Layers followed by a GlobalAveragePooling Layer 
plus the output Dense layer at the end. The Convolutional Layers are there to extract features from the image. 
The Pooling layers then reduce the complexity to prevent overfitting. They also take out the spatial information 
so that at the end only the information about the content remains. I've then included a GlobalAveragePooling Layer 
to take out more complexity. I've experimented with including a hidden Dense layer with 1000 nodes and a "relu" 
activation function before the final output layer but it didn't really make a big difference in train accuracy. 
In fact it did rather drive down the accuracy of the test set predictions so I've removed it from the model. 
The final Dense layer hass 133 nodes as this is the number of (dog-) classes that should be predicted. 
The activation function has to be softmax as we want to have probabilities for our predictions.
As we are making categorical predictions the loss function of the optimizer has to be "categorical_crossentropy". 
"rmsprop" is a good optimizer for classification tasks.

After training the model for 10 epochs it achieved a test accuracy of almost 4%. This could surely be improved by 
using more epochs for our training or getting (a lot) more images for training but it is better than random guessing.


* Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning) <br>
In this step I've used transfer learning to use the VGG-16 model pretrained on the ImageNet database without the 
final dense layers. The layers of the imported model have been frozen so the model can be used as input for our own model.
This model achieved a test accuracy of about 43%.

* Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning) <br>


* Step 6: Write the Algorithm <br>

* Step 7: Test the Algorithm <br>
 
  

### Results
Implementing the model and training it with our dataset only reached around 6% accuracy
Transfer Learning really useful; 

### Conclusion
problems with graph and session in flask; new Resnet version has a different output...
need to learn more about javascript to make the web experience more comfortable for the user...