'''
Это пример обучения модели на основе LSTM для распознавания тональности сообщения.
Обученная модель будет распознавать две тональности: позитивную и негативную.
Модель будет обучаться на корпусе данных IMDB - это размеченный корпус данных комментариев о фильмах.
'''

# импортируем функцию print из библиотеки future для совместимости двух версий python (python2 и python3)
from __future__ import print_function

# импортируем класс sequence для работы с последовательностями слов
from keras.preprocessing import sequence

# импортируем класс Sequential для создания каркаса модели
from keras.models import Sequential

# импортируем классы Dense, Embedding для создания слоев модели
from keras.layers import Dense, Embedding
# импортируем класс LSTM для создания LSTM слоя
from keras.layers import LSTM
# импортируем корпус данных imdb - это корпус диалогов из фильмов 
from keras.datasets import imdb
# imdb - это корпус данных отзывов о фильмах, размеченных для задачи распознавания тональности


max_features = 20000
maxlen = 80  # зададим максимальную длину строки
# мы будем обрезать все строки, число слов в которых больше maxlen
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# загруженный корпус данных - это последовательность из цифр, 
# каждая из которых соответствует определенному слову

print(x_train[1])
print(y_train[1])
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')

# # чтобы посмотреть словарь, с использованием которого составлены эти последовательности слов:
# word_to_id = imdb.get_word_index()
# # посмотрим, какому числу соответствует слово "happy":
# print(word_to_id['happy'])

# # конвертируем данные в формат 2D Numpy array (двумерного массива Numpy)
# # обрежем строки, длина которых превышает максимальную
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# # посмотрим на данные, которые получились в результате,
# # выведем первый объект массива
# print (x_train[0])

# # выведем данные о числе объектов и форме тензора
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)

# # теперь создадим модель

# print('Build model...')

# # сначала создадим объект класса Sequential - это каркас нашей будущей модели

# model = Sequential()

# # добавим слой для создания эмбедингов
# model.add(Embedding(max_features, 128))

# # добавим LSTM слой
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# # добавим полносвязный слой с сигмоидой в качестве функции активации
# model.add(Dense(1, activation='sigmoid'))

# # зададим параметры обучения модели:
# # функцию потерь: loss='binary_crossentropy'
# # алгоритм оптимизации: optimizer='adam'
# # метрику качества: metrics=['accuracy']
# # попробуйте задать другие параметры модели, например, поменять алгоритм оптимизации
# # почитать о том, какие бывают функции оптимизации можно тут: https://keras.io/optimizers/

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


# # обучим нашу модель
# print('Train...')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=15,
#           validation_data=(x_test, y_test))

# # выведем финальные метрики качества
# score, acc = model.evaluate(x_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)

# # сохраним полученную модель
# model.save('my_sent_model.h5')