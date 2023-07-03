#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import matplotlib.pyplot as plt

def getPointCloud(filename):
    file = open(filename)

    x = []
    y = []

    for line in file:
        line_list = line.split(' ')
        x_ = float(line_list[0])
        y_ = float(line_list[1])
        x.append(x_)
        y.append(y_)

    return x, y

def getLine(filename):
    file = open(filename)

    a = []
    b_x = []
    b_y = []
    c_x = []
    c_y = []
    d = []

    for line in file:
        if (line.find('line function') != -1):
            line_list = line.split(':')
            num_list = line_list[1].split(',')
            a.append([float(num_list[0]), float(num_list[1]), float(num_list[2]), float(num_list[3])])
        elif (line.find('closed points') != -1):
            line_list = line.split(':')
            num_list = line_list[1].split(',')
            b_x.append(float(num_list[0]))
            b_y.append(float(num_list[1]))
        elif (line.find('distributed point') != -1):
            line_list = line.split(':')
            num_list = line_list[1].split(',')
            c_x.append(float(num_list[0]))
            c_y.append(float(num_list[1]))
        elif (line.find('candidate1 line') != -1):
            line_list = line.split(':')
            num_list = line_list[1].split(',')
            d.append([float(num_list[0]),float(num_list[1]), float(num_list[2])])

    return a, [b_x, b_y], [c_x, c_y], d

if __name__ == '__main__':
    filename1 = '/home/dell/slam/LineSegmentFitting/data/merged_cloud_1.asc'
    filename2 = '/home/dell/slam/LineSegmentFitting/build/a.txt'
    x1, y1 = getPointCloud(filename1)
    a, b, c, d = getLine(filename2)
    k = [0.0261859,1,32.1193]
    n = [0.401378,0.915913,1.29134]
    
    plt.plot(x1, y1, 'co', label='point cloud', markersize=1)
    
    # for i in range(len(c)):
    #     plt.plot(c[0], c[1], 'mo', label='distributed points', markersize=1)
    for i in range(len(b)):
        plt.plot(b[0], b[1], 'go', label='closed points', markersize=1)

    for i in range(len(a)):
        plt.plot([a[i][0],a[i][2]], [a[i][1],a[i][3]], color = 'r', label='fitted lines')

    for i in range(len(d)):
        m = d[i]
        print(m)
        if m[1] == 0:
            plt.plot([-m[2]/m[0],-m[2]/m[0]], [-30,30], color = 'r', label='candidate lines')
        else:
            x = [7,22]
            plt.plot([x[0],x[1]], [(-m[0]*(x[0])-m[2])/m[1],(-m[0]*x[1]-m[2])/m[1]], color = 'r', label='candidate lines')

    # plt.plot([-30,30], [(-k[0]*(-30)-k[2])/k[1],(-k[0]*30-k[2])/k[1]], color = 'r', label='fitted lines')
    # plt.plot([30,(-n[0]*30-n[2])/n[1]], [45,(-n[0]*45+n[2])/n[1]], color = 'r', label='fitted lines')
    
    plt.xlim(-100,300)
    plt.ylim(-100,300)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True)

    plt.show()
