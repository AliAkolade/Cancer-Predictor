from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 23].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = Sequential()
model.add(Dense(250, input_dim=23, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5000, batch_size=25)

model.save('NeuralNet.hdf5')

accuracy = model.evaluate(X_test, y_test)
print('Accuracy = %.2f' % (accuracy*100))
