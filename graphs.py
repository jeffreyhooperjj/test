#!/usr/bin/python
import csv
import matplotlib.pyplot as plt 

frame = []
reward = []

with open('latest_rewards_7.csv', newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
    line_count = 1
    for row in csv_reader:
        if (line_count % 2 == 0):
            reward.append(float(row[0]))
        else:
            frame.append(float(row[0]))
        line_count += 1
    
# for i in range(len(frame)):
#     print("%s %s" % (frame[i], reward[i]))

# plotting the points  
plt.scatter(frame, reward) 
  
# naming the x axis 
plt.xlabel('Frame') 
# naming the y axis 
plt.ylabel('Reward') 
  
# giving a title to my graph 
plt.title('Frame vs Reward') 
  
# function to show the plot 
plt.show() 