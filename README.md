# NumberPredictor
Draw a number and it will predict what number you drew.


I am learning deep learning and this is essentially a modified version of an example model trained on the MNIST dataset that is on the keras website.

Link: https://keras.io/examples/vision/mnist_convnet/

I mostly just added a few layers to give the model a bit more depth because the model seemed to struggle with zeros and nines. I imagine this is due to the lack
of processing the image in my draw program. The original model I trained with fewer layers in the the MNIST-Model folder while the newer and more accurate model is in the 
model2 folder. 

Once I trained the model using number_model I made the draw.py program which is a simple little "game" using pygame. It pops up a screen and you can draw a number 0-9 and then hit the save 
button and it saves what is in the red box (your number). It then uses the model to predict what the number is. It became decently accurate but it can be easy to trick. 
It obviously does not do well with numbers that are not 0-9 and if the number is not relatively centered it can encounter errors as well.
