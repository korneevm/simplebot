import logging
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from cat_checker import img_has_cat
import settings

logging.basicConfig(level=logging.INFO, filename="bot.log", format='%(asctime)s - %(levelname)s - %(message)s')


def reply_to_start_command(bot, update):
    first_name = update.effective_user.first_name
    update.message.reply_text("Пришли фото котика {}".format(first_name))


def check_cat(bot, update):
    update.message.reply_text("Обрабатываю фото")
    photo_file = bot.getFile(update.message.photo[-1].file_id)
    filename = os.path.join('downloads', '{}.jpg'.format(photo_file.file_id))
    photo_file.download(filename)
    if img_has_cat(filename):
        update.message.reply_text("На картинке есть кошка, добавляю в библиотеку.")
        new_filename = os.path.join('cats', '{}.jpg'.format(photo_file.file_id))
        os.rename(filename, new_filename)
    else:
        os.remove(filename)
        update.message.reply_text("Тревога, кошка не обнаружена!")


def start_bot():
    my_bot = Updater(settings.TELEGRAM_API_KEY)

    dp = my_bot.dispatcher
    dp.add_handler(CommandHandler("start", reply_to_start_command))
    dp.add_handler(MessageHandler(Filters.photo, check_cat))

    my_bot.start_polling()
    my_bot.idle()


if __name__ == "__main__":
    start_bot()
