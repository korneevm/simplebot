# Импортируем нужные библиотеки
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

'''
Пример реализации Sequence to sequence при помощи библиотеки Keras
модель выполнена на уровне работы с буквами (character-level)

Задача этого примера - продемонстрировать как работает 
sec2sec модель на примере перевода коротких фраз с русского на английский. 
Перевод осуществляется побуквенно, то есть буква за буквой.

# Откуда можно скачать данные:
Пары фраз на различных языках
http://www.manythings.org/anki/
# Ссылки на статьи и материалы по теме:
- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
- другие примеры использования библиотеки Keras для задач машинного обучения:
    https://github.com/keras-team/keras/blob/master/examples/
'''




batch_size = 64  # Зададим размер батча
epochs = 100  # Зададим число эпох
latent_dim = 256  # Размерность вектора (векторов) состояния
num_samples = 10000  # Число объектов, на которых мы будем обучать нейронную сеть
# Зададим путь до файла с обучающими данными
data_path = 'rus.txt'





# Переведем текстовые данные в вектора:


input_texts = []
target_texts = []

# Создадим словарь символов

input_characters = set()
target_characters = set()
# создаем множества input_characters и target_characters
# множество в python - "контейнер", содержащий не повторяющиеся элементы в случайном порядке

# откроем файл с данными
# разобьем данные по знаку переноса строки и запишем в список
lines = open(data_path).read().split('\n')

# давайте посмотрим на наши данные
# выведем один из элементов списка

# print (lines[0])

# : - это слайсы, [0:10] - обозначает "взять элементы с 0 по 10-й"
# при такой записи 0 часто опускают
# min(num_samples, len(lines) - 1) - выбрать минимальное из двух значений:  num_samples и len(lines) - 1)
# len(lines) - 1) - это число элементов в списке lines минус 1. Единицу нужно вычесть, 
# так как значение, подставляемое в lines[] - это максимальный индекс, до которого мы дойдем, перебирая список
# индексация начинается с 0, следовательно, чтобы найти максимальный индекс 
# из значения длины списка нужно вычесть 1

for line in lines[: min(num_samples, len(lines) - 1)]:

    # разделим каждую строчку на 2 по знаку табуляции
    input_text, target_text = line.split('\t')
    # "tab" - будет обозначать начало последовательности для "целей"
    # "\n" - будет обозначать конец последовательности
    target_text = '\t' + target_text + '\n'
    
    # запишем фразы на каждом из языков в соответствующие списки
    input_texts.append(input_text)
    target_texts.append(target_text)

    # составим словарь символов для input text:
    # (это символы языка, с которого будем переводить)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)

    # составим словарь символов для target_text:
    # (это символы языка, с на который будем переводить)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


# # Проверим, что у нас получилось:
# # фраза на английском:
# print(input_texts[0])
# # фраза на русском:
# print(target_texts[0])
# # словарь английских символов
# print(input_characters)
# # словарь русских символов
# print(target_characters)

# Отсортируем полученные словари символов
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
# выведем размеры словарей
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
# посчитаем максимальную длину строки в списках строк
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# выведем все эти данные

# print('Number of samples:', len(input_texts))
# print('Number of unique input tokens:', num_encoder_tokens)
# print('Number of unique output tokens:', num_decoder_tokens)
# print('Max sequence length for inputs:', max_encoder_seq_length)
# print('Max sequence length for outputs:', max_decoder_seq_length)

# добавим индексы к символам языка и запишем эти символы с соответствующими индексами в словарь
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])





# np.zeros - вернет массив numpy, заполненный нулями
# в скобках задаем форму этого массива

# для обучения энкодера:
# первое измерение - это len(input_texts) число фраз этого языка
# второе измерение - max_encoder_seq_length - максимальная длина строки языка, с которого переводим
# третье измерение - num_encoder_tokens - число символов с словаре соответствующего языка

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

# те же параметры зададим для обучения декодера:

decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# для обучения декодера-переводчика:

decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')



#  zip - функция, которая позволяет слить списки парами:
"""
a = [1,2]
b = [3,4]
print zip(a,b)
[(1, 3), (2, 4)]
"""
# то есть мы будем работать построчно с каждым списком
# закодируем данные о тексте в массивы numpy в виде one-hot векторов


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.

    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.


        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# # Подготовка данных закончена!





# Теперь создадим архитектуру модели:

# Сначала зададим модель-encoder (для обрабоки фраз текста-источника):

# создадим объект класса Input - так мы определяем форму входящей матрицы
# None, переданное в качестве параметра, позволяет принимать тензоры любой размерности по этой координате

encoder_inputs = Input(
    shape=(None, 
        num_encoder_tokens
        )
    )

# зададим LSTM слой в качестве энкодера
# первый параметр latent_dim - это размер вектора скрытого состояния, 
# который будет выдавать рекуррентный слой
# функции активации заданы по умолчанию:
# activation = 'tanh' 
# recurrent_activation = 'hard_sigmoid' 
# мы не будем их менять, так же как и другие параметры
# подробнее о параметрах функции тут: https://keras.io/layers/recurrent/#lstm
# return_state = True - LSTM слой будет возвращать скрытое состояние 
# вместе с output на последнем шаге последовательности

encoder = LSTM(latent_dim, return_state=True)

# созраним выходные данные из encoder слоя в отдельные переменные:
# encoder_outputs - это выходные данные - "прогноз следующего слова" 
# state_h, state_c - это векторы скрытого состояния, их мы и будем использовать

encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# в переменную encoder_states запишем только векторы скрытого состояния
encoder_states = [state_h, state_c]


# Теперь зададим модель-decoder (для обрабоки фраз текста-цели):

# сеть декодер будет использвоать `encoder_states` в качестве своего начального состояния,
# пропускать его через LSTM слой и возвращать текст

# так же как в случае encoder сначала задаем форму тензора,
# num_decoder_tokens - размер словаря символов целевого языка
decoder_inputs = Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.

# задаем LSTM слой
decoder_lstm = LSTM(
    latent_dim, 
    return_sequences=True, 
    return_state=True)

# Это присвоение переменных. 
# Функция возвращает кортеж из тех значений, 
# значения которых пишутся в именованные переменные или игнорируются, если _
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

# нам понадобится еще один полносвязный слой:
# в качестве первого параметра передаем размер выходного вектора из этого слоя
# размер выходного вектора равен num_decoder_tokens
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)




# Теперь объединим слои в одну модель
# Модель будет получать на вход `encoder_input_data` & `decoder_input_data` 
# и конвертировать их в `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Здадаим параметры обучения модели
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# обучим модель
# validation_split=0.2 задает параметры разбиения данных на тренировочным и валидационные

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Сохраним модель

model.save('s2s.h5')



# Теперь обучим модель вывода текста (inference model)

# Что нам предстоит сделать:

# 1) Закодируем инпут и сгенерируем начальный вектор состояния
# 2) Зпустим декодер и трансформируем вектор состояния в вектор символа,
# первым символом будет символ начала строки.
# Запустим второй декодер и сгенерируем первый символ фразы на целевом языке
# 3) Повторим все, начиная с шага 1, во второй декодер будем подавать весь набор символов, 
# сгенерированных на предыдущих шагах


# Создадим Inference model

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state = decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# Создадим словарь, с ключами в виде индексов и символами

reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# Предскажем следующий символ при помощи Inference model

def decode_sequence(input_seq):
    # Закодируем фразу в виде вектора состояния
    states_value = encoder_model.predict(input_seq)

    # Создадим пустую фразу-цель, размер которой равен 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Инициируем первый символ в целевой последовательности символом \t
    target_seq[0, 0, target_token_index['\t']] = 1.

    # В цикле предскажем каждую следующую букву фразы
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        # предсказание векторов состояния и output
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # получаем следующую букву в последовательности, используя output
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        # добавляем полученную букву к последовательности
        decoded_sentence += sampled_char

        # Заканчиваем цикл, если достигли максимальной длины фразы или стоп-символа
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Обнулим целевую последовательность
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Запишем предсказанный на текущем шаге индекс в целевую последовательность
        target_seq[0, 0, sampled_token_index] = 1.

        # Обновим векторы состояния
        states_value = [h, c]
        # передадим новые векторы состояния и предсказанный символ на следующую итерацию цикла

    return decoded_sentence


for seq_index in range(100):
    # Возьмем 100 первых фраз и переведем их
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)