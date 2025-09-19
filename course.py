import matplotlib.pyplot as plt
import numpy as np

def plot(vecs):
    colors = 'r','b','g','y'
    i=0
    for vec in vecs:

       
         plt.quiver(vec[0],vec[1],vec[2],vec[3], scale_units="xy", angles='xy',scale=1,color= colors[i%len(colors)])
         i+=1
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.show()
plot([(0,0,3,4),(0,0,-3,4),(0,1,4,2)])