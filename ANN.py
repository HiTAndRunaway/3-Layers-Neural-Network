# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy.special as sci
import matplotlib.pyplot as plt
#三层的神经网络（不带偏置的三层感知机）
class NeuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inode = inputnodes
        self.hnode = hiddennodes
        self.onode = outputnodes
        self.lr = learningrate
        
        #self.wih = np.random.rand(self.hnode,self.inode) - 0.5
        #self.who = np.random.rand(self.onode,self.hnode) - 0.5
        #输入层到隐藏层的权重和隐藏层到输出层的权重
        self.wih = np.random.normal(0.0,pow(self.hnode,-0.5),(self.hnode,self.inode))
        self.who = np.random.normal(0.0,pow(self.onode,-0.5),(self.onode,self.hnode))
        #激活函数sigmoid
        self.activation_function = lambda x : sci.expit(x)
    
    def train(self,inputs_list,targets_list):
        #一条向量转置（T），维度（ndmin）
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = hidden_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        #输出层误差
        output_errors = targets - final_outputs
        #隐藏层误差
        hidden_errors = np.dot(self.who.T,output_errors)
        #矩阵转置（np.transpose）
        #梯度下降 得到新的权重 反向传播
        self.who -= self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs),np.transpose(hidden_outputs))
        self.wih -= self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs),np.transpose(inputs))
    
    def query(self,inputs):
        #输入层到隐藏层
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #隐藏层到输出层
        final_inputs = hidden_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learningrate = 0.3

n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learningrate)

data_file = open("E:\mnist\mnist_train.csv","r")
data_list = data_file.readlines()
data_file.close()
epochs = 2
for e in range(epochs):
    for record in data_list:

        all_values = record.split(',')
        #image_array = np.asfarray(all_values[1:]).reshape(28,28)
        #plt.imshow(image_array,cmap = 'Greys',interpolation = 'None')
        #plt.show()

        input = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01

        onodes = 10
        targets = np.zeros(onodes) + 0.01

        targets[int(all_values[0])] = 0.99

        n.train(input,targets)

test_data_file = open("E:\mnist\mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    correct_label = int(all_values[0])

    if correct_label == label:
        scorecard.append(1)
    else:
        scorecard.append(0)
#print(scorecard)
arr = np.asarray(scorecard)
print("performance = ",arr.sum() / arr.size)
