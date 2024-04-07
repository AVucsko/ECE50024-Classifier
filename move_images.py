import csv
import os
import cv2

# Initialize classes
classes = [0,'']*100
class_file = open('category.csv')
csvreader = csv.reader(class_file)
idx = 0
for row in csvreader:
    if idx > 0 and idx <= 100:
        classes[idx-1] = row
    idx = idx + 1
#print(classes)

for i in range(0,100):
    os.mkdir(str('sep_faces_2/'+str(classes[i])))

# Initialize answers
train_file = open('train.csv')
csvreader = csv.reader(train_file)
answers = [""]*(69540)
indices = [""]*(69540)
idx = 0
for row in csvreader:
    if idx > 0 and idx < 69540:
        indices[idx-1] = row[1]
        answers[idx-1] = row[2]
    idx = idx + 1

for filename in os.listdir('combined_faces'):
    if filename.endswith('.jpg'):
        for x in range(0, 69540):
            #print(indices[x], answers[x], filename)
            if str(indices[x]) == str(filename):
                print(filename, answers[x])
                img = cv2.imread('combined_faces/' + filename)
                img = cv2.resize(img, (180,180))
                cv2.imwrite('separated_faces/' + str(answers[x]) + '/' + str(filename), img)

for filename in os.listdir('test_faces'):
    if filename.endswith('.jpg'):
        img = cv2.imread('test_faces/' + filename)
        img = cv2.resize(img, (180,180))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite('test_faces/' + str(filename), img)