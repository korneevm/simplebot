import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ReplyKeyboardMarkup, KeyboardButton

from weather import get_weather_by_city
import settings

logging.basicConfig(level=logging.INFO, filename="bot.log", format='%(asctime)s - %(levelname)s - %(message)s')


def reply_to_start_command(bot, update):
    first_name = update.effective_user.first_name
    last_name = update.effective_user.last_name
    text = "Привет, {} {}! Я бот, который подскажет погоду. Введите команду /weather Город".format(first_name, last_name)
    update.message.reply_text(text)


def get_weather(bot, update, args):
    try:
        city = args[0]
        weather = get_weather_by_city(args[0])
        if weather:
            weather_text = """
Температура: {}
Давление: {}
Влажность: {}""".format(weather['main']['temp'], weather['main']['pressure'], weather['main']['humidity'])
            update.message.reply_text(weather_text)
        else:
            update.message.reply_text('Город не найден, попробуйте снова')
    except(KeyError):
        update.message.reply_text('Пожалуйста, введите название города')

def start_bot():
    my_bot = Updater(settings.TELEGRAM_API_KEY)

    dp = my_bot.dispatcher
    dp.add_handler(CommandHandler("start", reply_to_start_command))
    dp.add_handler(CommandHandler("weather", get_weather, pass_args=True))

    my_bot.start_polling()
    my_bot.idle()


if __name__ == "__main__":
    start_bot()
