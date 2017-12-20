# импортируем keras, чтобы скачать данные и создать модель нейронной сети
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils

# импортируем numpy для работы с массивами
import numpy as np

# импортируем os для работы с папками и файлами
import os

# импортируем scipy для сохранения данных в виде картинок
import scipy.misc

# этой строчкой мы объявляем, что видео-карта при обучении использоваться не будет
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# эта функция задает параметры генерации случайных чисел
# их важно задавать одинаково для всех экспериментов
# разница в генерации случайных весов не должна влиять на полученную в результате модель
np.random.seed(0)

# загрузим трейн и тест датасеты
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Чтобы иметь возможность посмотреть на данные глазами,
# давайте сохраним несколько объектов в виде картинок

directory = 'mnist_images'
im_num = 20

def to_jpg(directory, im_num):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(im_num):
        scipy.misc.imsave('{}/image_{}.jpg'.format(directory, i), X_test[i])
    print('images created in {} "mnist_images", enjoy!'.format(directory))


to_jpg('mnist_images', im_num)

# Подготовка данных:

# наши данные импортировались в формате numpy.ndarray
# давайте посмотрим на тип данных
print (type(X_test))

# давайте посмотрим на данные
print (X_test[0])
print (y_test[0])

# давайте посмотрим на форму массива
print (X_test.shape)
print (y_test.shape)


# конвертируем каждую картинку из матрицы в вектор:
# развернем матрицу 28*28 в вертор, размером 784

# для начала вычислим размер результарующего вектора,
# для этого умножим одно измерение матрицы на другое

# функция shape() позволяет отобразить форму numpy массива
num_pixels = X_train.shape[1] * X_train.shape[2]

# функция reshape() позволяет изменить эту форму
# в качестве аргумента функция принимает форму будущего массива
# в данном случае в качестве формы мы передаем:
# по X - число объектов, то есть X_train.shape[0] = 60000
# по Y - размер вектора num_pixels = 28*28 = 784
# таким образом у нас получится массив векторов, в котором будет 60000 объектов,
# размер каждого объекта 784
# с этим массивом (матрицей) и будем работать
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# давайте посмотрим на получившиеся объекты

print(X_train[0])
print(X_test[0])

# и на их форму

print(X_train[0].shape)
print(X_test[0].shape)

# теперь нормализуем данные, т.е. разделим все на максимальное значение 255
# до нормализации значения были от 0 до 255
# после нормализации от 0 до 1
X_train = X_train / 255
X_test = X_test / 255

# давайте посмотрим на получившиеся данные

print(X_train[0])
print(X_test[0])


# теперь подготовим к работе лейблы - то есть Y
# давайте сначал посмотрим на лейблы

print(y_test)
print(y_test[0])

# для лейблов необходимо использовать one hot encoding
# для этого есть специальная функция np_utils.to_categorical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# посмотрим на то, что получилось

print(y_test)
print(y_test[0])

# запишем число классов в отдельную переменную.
# она нам пригодится в дальнейшем
num_classes = y_test.shape[1]


# Данные готовы к использованию!
# Теперь создадим модель


# Создаем модель как объект класса Sequential
# Sequential model - это модель с последовательной передачей данных из слоя в слой
model = Sequential()


# при помощи функции add будем добавлять слои в нашу модель

model.add(
    Dense(
        num_pixels,
        input_dim=num_pixels,
        kernel_initializer='normal',
        activation='relu'
    )
)

# Dence - это объект, при помощи которого мы будем создавать слои
# первый параметр - это число нейронов.
# В нашем случае оно равно num_pixels = 784
# вы можете попробовать задать другое число нейронов
# Dense(784) - это полносвязный слой с 784 нейронами
# функция активации для первого слоя - RELU
# kernel_initializer='normal' - это настройки ядра по умолчанию

# теперь добавим выходной слой в нашу модель:

model.add(
    Dense(
        num_classes,
        kernel_initializer='normal',
        activation='softmax'
    )
)
# добавляем такой же полносвязный слой (fully-connected layer)
# размер выходного слоя равен 10 нейронам (это число классов)
# в качестве функции активации используется softmax

# Соберем модель при помощи функции model.compile
# нам необходимо задать:
# функцию лосса loss='categorical_crossentropy'
# функцию оптимизации для градиентоного спуска optimizer='adam'
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# тренировать модель будем при помощи функции model.fit()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# в функцию необходимо передать:
# тренировочный датасет X_train,
# лейблы тренировочного датасета y_train
# валидационный датасет X_test,
# лейблы валидационного датасета y_test
# так же необходимо задать число объектов в батче batch_size=200
# и число эпох: epochs=10
# verbose = 2 это режим отображения прогресса:
# 0 = не отображать,
# 1 = progress bar,
# 2 = одна строка с информацией после каждой эпохи

# после обучения выведем финальные данные о полученной модели:
# выведем получившуюся ошибку - процент неправильных ответов при предсказании
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

# и сохраним модель в отдельный файл
model.save('my_model.h5')
