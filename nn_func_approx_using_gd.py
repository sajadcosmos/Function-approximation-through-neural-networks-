import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# Input data
# define the dataset
# x = np.array(np.random.rand(1000))
# y = np.asarray(np.sin(x))
# # reshape arrays into into rows and cols
# x = x.reshape((len(x), 1))
# y = y.reshape((len(y), 1))
# # separately scale the input and output variables
# scale_x = MinMaxScaler()
# x = scale_x.fit_transform(x)
# scale_y = MinMaxScaler()
# y = scale_y.fit_transform(y)


# Generate Data
def cubic(x):
    return x**3 + x**2 - x - 1


def get_cubic_data(start, end, step_size):
    X = np.arange(start, end, step_size)
    X.shape = (len(X), 1)
    y = np.array([cubic(X[i]) for i in range(len(X))])
    y.shape = (len(y), 1)
    return X, y


x, y = get_cubic_data(-2, 2, .1)
#
# scale_x = MinMaxScaler()
# x = scale_x.fit_transform(x)
# scale_y = MinMaxScaler()
# y = scale_y.fit_transform(y)


# Train Model
X, y = get_cubic_data(-2, 2, .1)
# Build the model.
model = Sequential([
  Dense(100, input_dim=1, activation='relu', kernel_initializer='he_uniform'),
  # Dense(100, activation='relu', kernel_initializer='he_uniform'),
  # Dense(100, activation='relu', kernel_initializer='he_uniform'),
  Dense(1),
])


# Compile the model.
model.compile(
  optimizer='adam',
  loss='mse',
)


# Train the model.
model.fit(
  x,
  y,
  epochs=1000,
  batch_size=15,
)


#test_data = np.array([np.random.rand(20)]).reshape(20,1)
y_actual = y
yhat = model.predict(x)
# inverse transforms
# x_plot = scale_x.inverse_transform(x)
# y_plot = scale_y.inverse_transform(y)
# yhat_plot = scale_y.inverse_transform(yhat)
from matplotlib import pyplot
# plot x vs y
pyplot.plot(y_actual, label='Actual')
# plot x vs yhat
pyplot.plot(yhat, label='Predicted')
pyplot.title('Input (x) versus Output (y) (Using Gradient Descent)')
pyplot.xlabel('Input Variable (x)')
pyplot.ylabel('Output Variable (y)')
pyplot.legend()
pyplot.show()