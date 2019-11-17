import numpy as np

class sw:
    def __init__(self):
        self.value = np.array([1,2,3,4,5,-1,-2,-3,0])

    def relu(self):
        #self.value=max(self.value,[0]*len(self.value))
        return np.maximum(self.value,0)
        
        



W=sw()
print(W.relu())