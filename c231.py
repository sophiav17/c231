from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt('diabetes_dataset.csv', delimiter = ',')

x = dataset[:, 0:8]
y = loadtxt[:, 8]
print('Value of X are: ', x)
print('Value of Y are: ', y)


model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()