#Imports

import pygame
import pygame.freetype
import cv2
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import load_model

pygame.init() #initialize the pygame, will not do anything without this first


#Set size of the screen that the game will be displayed on
size = [400, 400] 
screen = pygame.display.set_mode(size)

#Sets the text that appears at the top of the window and sets the font for future text
pygame.display.set_caption("Draw a number 0-9 in the box!")
FONT = pygame.freetype.SysFont("Times New Romans", 12)

#Booleans for later use and initializes clock for the pygame to run at different speeds if wanted using clock.tick(x)
done = False
drawing = False
clock = pygame.time.Clock()
 
pos = [0, 0] #Initializes a variable for the mouse location starting it in the top left corner
screen.fill((0, 0, 0)) #Fills the screen with a black background
pygame.draw.rect(screen, (150, 0, 0), (30, 30, 340, 340), 2) #draws the big red square to be drawn in
pygame.draw.rect(screen, (170, 170, 170), (323, 373, 75, 25), 0, 3) #draws the save "button"
FONT.render_to(screen, (350, 382), "Save", (0, 0, 0)) #renders text onto a location, in this case on the save "button"

#Function to load and predict a number from an image
def predict(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28))

    #CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    with tensorflow.device("cpu:0"):
        model = load_model("MNIST/model2")
        #img = img / 255
        print("[INFO] Predicting....")
        img = np.expand_dims(img, axis=-1)
        probs = model.predict(np.expand_dims(img, axis=0))[0]
        prediction = probs.argmax(axis=0)

        print("You drew the number ", prediction)
        exit()

#function to save the number that was drawn as a .jpg file
def save():
    rect = pygame.Rect(32, 32, 338, 338)
    sub = screen.subsurface(rect)
    print("[INFO] Saving....")
    pygame.image.save(sub, "MNIST/input.jpg")
    predict("MNIST/input.jpg")

#game loop
while not done:

    for event in pygame.event.get(): #polls for any event
        if event.type == pygame.QUIT: #if the window is closed, quit the game
            done = True
        elif event.type == pygame.MOUSEBUTTONDOWN: #if the mouse is pressed
            if pos[0] > 323 and pos[1] > 373: #checking if save is being clicked
                save()
            else:
                drawing = True #if the game is not being saved, draws while the mouse is still pressed
        elif event.type == pygame.MOUSEBUTTONUP: #once mouse button is released
            drawing = False #stop drawing

    if drawing:
        pos = pygame.mouse.get_pos() #gets the mouse location
        if pos[0] > 370 or pos[0] < 30 or pos[1] > 370 or pos[1] < 30: #checks if it in the red square
            continue
        else:
            pygame.draw.circle(screen, (255, 255, 255), (pos[0], pos[1]), 5) #shape that is drawn, it is a very high framerate so i used a circle to simulate a line, could use a rectangle or other

    pygame.display.flip()

pygame.quit()
