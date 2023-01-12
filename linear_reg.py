import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

list_x = []
list_y = []
num  = 1000


while(num):
    list_x.append(random.randint(0.0,100.0))
    list_y.append(random.randint(0.0,100.0))
    num -= 1
    
list_x = np.asarray(list_x,dtype='int')
list_y = np.asarray(list_y,dtype='int')

train = 800
test = 200
x_train,x_test,y_train,y_test = list_x[:train],list_x[:test],list_y[:train],list_x[:test]

def accuracy(y_pred,y_test):
    print("mae :",np.abs(y_test-y_pred).mean())
    print("rmse :",(((y_test-y_pred)**2).mean())**0.5)
    
def gradient_descent(m_now, b_now, x, y, L):
     m_gradient = 0
     b_gradient = 0
     
     n = 1000
     
     for i in range(n):
        x = list_x[i]
        y = list_y[i]
         
        m_gradient += (-2/n) * x * (y - (m_now * x + b_now))
        b_gradient += (-2/n) * (y - (m_now * x + b_now))
         
     m = m_now - m_gradient * L
     b = b_now - b_gradient * L
     
     return m, b
 
m = 0
b = 0
L = 0.0001
epochs = 300

for i in range(epochs):
    if i%50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, list_x, list_y, L)
    
print(m, b)
y_pred = m*x_test + b
accuracy(y_pred,y_test)
plt.scatter(list_x, list_y, color='pink')
plt.plot(list(range(0,100)), [m *x + b for x in range(0,100)], color = 'black')



from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))

c = reg.intercept_[0]
m = reg.coef_[0,0]

y_pred = reg.predict(x_test.reshape(-1,1))

accuracy(y_test,y_pred)
plt.scatter(list_x,list_y,c="r")
plt.plot(x_test,y_pred,c="b")


