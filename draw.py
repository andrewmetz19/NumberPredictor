import pygame
import pygame.freetype
import cv2
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import load_model

pygame.init()

size = [400, 400]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Draw a number 0-9 in the box!")
FONT = pygame.freetype.SysFont("Times New Romans", 12)

done = False
drawing = False
clock = pygame.time.Clock()

pos = [0, 0]
screen.fill((0, 0, 0))
pygame.draw.rect(screen, (150, 0, 0), (30, 30, 340, 340), 2)
pygame.draw.rect(screen, (170, 170, 170), (323, 373, 75, 25), 0, 3)
FONT.render_to(screen, (350, 382), "Save", (0, 0, 0))


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


def save():
    rect = pygame.Rect(32, 32, 338, 338)
    sub = screen.subsurface(rect)
    print("[INFO] Saving....")
    pygame.image.save(sub, "MNIST/input.jpg")
    predict("MNIST/input.jpg")


while not done:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if pos[0] > 323 and pos[1] > 373:
                save()
            else:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

    if drawing:
        pos = pygame.mouse.get_pos()
        if pos[0] > 370 or pos[0] < 30 or pos[1] > 370 or pos[1] < 30:
            continue
        else:
            pygame.draw.circle(screen, (255, 255, 255), (pos[0], pos[1]), 5)

    pygame.display.flip()

pygame.quit()
