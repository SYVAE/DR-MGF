# import torch
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    #0.1--100
    temperature=0.1
    iteration=100

    for i in range(0,100):
        temperature+=0.1
        x=np.linspace(-10,10,num=100)
        y=1.0/(1+np.exp(-temperature*x))
        plt.figure(1)
        plt.plot(x,y,'-')
        plt.pause(0.5)