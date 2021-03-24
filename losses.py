#!/usr/bin/python
import csv
import matplotlib.pyplot as plt 

frame = []
losses = []

with open('losses_7.csv', newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
    line_count = 1
    for row in csv_reader:
        if (line_count % 2 == 0):
            losses.append(float(row[0]))
        else:
            frame.append(float(row[0]))
        line_count += 1
    
# for i in range(len(frame)):
#     print("%s %s" % (frame[i], reward[i]))

# plotting the points  
plt.scatter(frame, losses) 
  
# naming the x axis 
plt.xlabel('Frame') 
# naming the y axis 
plt.ylabel('Loss') 
  
# giving a title to my graph 
plt.title('Frame vs Loss') 
  
# function to show the plot 
plt.show() 