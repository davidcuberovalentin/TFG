import matplotlib.pyplot as plt
from PIL import Image

def getPRF(detection_image):
    # Import detection image and ideal mask from directory:
    ideal_mask = Image.open("Ideal_Mask.png") #Blood pixels
    #ideal_mask = Image.open("Ideal_Mushroom.png")  #Mushroom pixels
    # Extracting the width and height of the image:
    width, height = ideal_mask.size
    TP = TN = FP = FN = 0
    # Comparing ideal mask with detection binarized image
    for i in range(0, width):
        for j in range(0, height):
            # Getting the RGB pixel value.
            grayscale_d = detection_image[j, i]
            grayscale_i = round(ideal_mask.getpixel((i, j)) / 255) #Blood mask
            # r, g, b = ideal_mask.getpixel((i,j)) #Mushroom mask
            #grayscale_i = round((0.299*r + 0.587*g + 0.114*b) / 255) #Mushroom mask
            if grayscale_d != 1 and grayscale_d != 0:
                print(grayscale_d, i, j)
            # Computing TP, TN, FP, FN
            if grayscale_d == 1 and grayscale_i == 1:
                TP += 1
            elif grayscale_d == 0 and grayscale_i == 0:
                TN += 1
            elif grayscale_d == 1 and grayscale_i == 0:
                FP += 1
            else:
                FN += 1
    # Computing Precision, Recall/TPR, F-score and FPR
    if TP == 0 and FP == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    FPR = FP / (FP + TN)
    print(Precision, Recall)
    print(Recall, FPR)
    if Recall == 0 and Precision == 0:
        F1_score = 0
    else:
        F1_score = (2 * Precision * Recall) / (Recall + Precision)
    print(F1_score)
    return Precision, Recall, F1_score, FPR

if __name__ == "__main__":
    getPRF(0)
