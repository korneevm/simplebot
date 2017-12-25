# Импортируем нужные библиотеки:

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

# Keras backend - это библиотека для работы с тензорами и совершения операций над ними
# Мы будем использовать ее для работы с данными (тензорами изображений)


# зададим параметры обучения нейронной сети:
batch_size = 128
num_classes = 10
epochs = 12

# зададим размеры входящи изображений
img_rows, img_cols = 28, 28

# загрузим данные для обучения нейронной сети: x_train, y_train
# загружим даныне для оценки качества обученной модели: x_test, y_test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# давайте посмотрим на форму массива данных, которые мы загрузили
# для этого напечатаем x_train.shape 
print(x_train.shape)


# K.image_data_format() - значение соответствует выбранной конвенции для работы с тензорами
# 'channels_first' - означает, что каналы будут перед числом строк и столбцов
# 'channels_last' - означает, что каналы будут после числа строк и столбцов
# по умолчанию выбрано значение 'channels_last'
# это можно проверить, напечатав такую строку:

print (K.image_data_format())

# переформатируем наши данные в соответствии с выбранной конвенцией

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# Зададим нужный тип данных - float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Нормализуем данные
x_train /= 255
x_test /= 255

# еще раз посмотрим на форму получившегося тензора:

print('x_train shape:', x_train.shape)

# выведем число объектов в обучающий и тестовой выборках:
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# используем one-hot encoding для конвертации лейблов классов из чисел в вектора
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# теперь создадим модель
# сначала создадим объект класса Sequential
# Sequential - означает, что наша модель будте последовательно передавать данные из слоя в слой

model = Sequential()

# добавим в модель первый сверточный слой:
# Conv2D - сверточный слой, сворачивать будем двумерную матрицу, поэтому 2D

# первый аргумент filters ( = 32) задает число фильтров, при помощи которых мы будем сворачивать изображение
# один фильтр эквивалентен одному нейрону (в полносвязной сети)
# таким образом число фильтров эквивалентно числу нейронов
# число features map, которые мы молучим на выход из первого слоя будет равно числу использованных фильтров

# второй аргумент - kernel_size задает параметры фильтра, 
# в нашем случае сворачивать будем при помощи фильтра (свертки) 3 на 3

# третий аргумент: в качестве функции активации будем использовать ReLU

# четвертый аргумент: input_shape - размер тензора, который будем подавать на вход


model.add(
	Conv2D(32, 
		kernel_size=(3, 3),
		activation='relu',
		input_shape=input_shape)
	)

# добавим еще один сверточный слой,
# содержащий 64 фильтра размером 3 на 3
# с функцией активации ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))
# добавим макс пулигн слой, с фильтром размерностью 2 на 2
model.add(MaxPooling2D(pool_size=(2, 2)))
# добавим дроп-аут, исключать будем 25% значений случайным образом
model.add(Dropout(0.25))

# после этого конвертируем полученные данные в одномерный вектор
model.add(Flatten())
# после этого добавим два полносвязных слоя: 
# один размерностью 128, в качестве функции активации - ReLU 
model.add(Dense(128, activation='relu'))
# добавим дроп-аут, будем исключать 50% данных случайным образом
model.add(Dropout(0.5))
# еще одни полносвязный слой, с Softmax в качестве функции активации
model.add(Dense(num_classes, activation='softmax'))


# при помощи model.compile зададим функцию потерь, алгоритм оптимизации, и метрику для имерения точности
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



# обучим модель: передадим обучающие данные, лейблы обучающих данных, 
# зададим размер батча, число эпох, выберем решим отображения процесса обучения (verbose=1)
# передадим тестовые данные для подсчета метрики качества

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# выведим метрики качества обученной модели
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# сохраним обученную модель
model.save('my_conv_model.h5')