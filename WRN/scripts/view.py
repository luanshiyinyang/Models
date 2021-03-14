"""
Author: Zhou Chen
Date: 2020/4/16
Desc: desc
"""
import pickle
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


with open('his.pkl', 'rb') as f:
    his = pickle.load(f)

plt.figure(figsize=(12, 6))
plt.plot(list(range(len(his['train_loss']))), his['train_loss'], c='r', marker='*', label="training loss")
plt.plot(list(range(len(his['valid_loss']))), his['valid_loss'], c='b', marker='.', label="validation loss")
plt.legend(loc=0)
plt.savefig('../assets/his.png')
plt.show()