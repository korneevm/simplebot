# импортируем os для работы с папками и файлами
import os
# импортируем keras, чтобы загрузить готовую модель
# и работать с ней
from keras.models import load_model
# импортируем numpy для работы с массивами
import numpy as np
# импортируем модуль Image из библиотеки PIL для чтения картинок
from PIL import Image

# импортируем модуль misc из библиотеки scipy для сохранения картинок
import scipy.misc

# Напишем функцию, которая будет получать картинку из файла
# и возвращать число, которое написано на этой картинке


def get_num(file_name):
    # откроем выбранный файл и сохраним картинку
    chosen_pic = Image.open(file_name)
    chosen_pic.load()

    # сделаем картинку черно-белой (на случай, если нам передали цветную)
    chosen_pic = chosen_pic.convert('L')

    # приведем картинку к нужному размеру при помощи функции resize()
    size = 28, 28
    chosen_pic = chosen_pic.resize(size)

    # конвертируем данные в формат массива numpy
    data = np.asarray(chosen_pic, dtype="float32")

    # сохраним получившиеся данные в виде картинки - проверим, все ли в порядке
    path, file = os.path.split(file_name)
    reshaped = os.path.join(path, "resaped_{}".format(file))
    scipy.misc.imsave(reshaped, data)

    # переформатируем данные в понятный модели формат:
    # нам нужно получить массив векторов размером 784
    num_pixels = data.shape[0] ** 2
    # первое измерение тензора равно 1, так как у нас одна картинка
    reshaped_data = data.reshape(1, num_pixels).astype('float32')

    # загрузаем модель
    model = load_model('my_model.h5')

    # пропускаем данные через модель - предсказываем ответ
    prediction = model.predict(reshaped_data)
    # функция model.predict возвращает массив массивов (тензор),
    # содержащий предсказание

    print(prediction)

    # мы передаем только один объект в функцию,
    # следовательно нам нужно только первое предсказание
    # теперь переменная prediction содержит список с предсказаниями,
    # соответствующий one-hot вектору лейблов, который мы использовали для тренировки модели
    prediction = model.predict(reshaped_data)[0]

    # функция np.amax возвращает максимальное значение в списке
    max_value = np.amax(prediction)

    # определим индекс этого максимального значения - это и будет наш ответ
    # используем для этого цикл - будем перебирать массив предсказаний,
    # пока не найдем нужное
    for idx, val in enumerate(prediction):
        if val == max_value:
            result = idx
    # вернем найденное значени, чтобы наша функция работала правильно,
    # обернем возвращаеме значение в try except
    try:
        return result
    except (IndexError):
        return False


if __name__ == "__main__":
    print(get_num('test_images/image1.jpg'))
