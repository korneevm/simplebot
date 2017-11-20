from glob import glob1
import logging
import random
import os

from emoji import emojize
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import settings

avatars = [':smiley_cat:', ':smiling_imp:', ':panda_face:', ':dog:']

logging.basicConfig(level=logging.INFO, filename="bot.log")


def get_avatar(user_data):
    if user_data.get('avatar'):
        return user_data.get('avatar')
    else:
        user_data['avatar'] = emojize(random.choice(avatars), use_aliases=True)
        return user_data['avatar']


def reply_to_start_command(bot, update, user_data):
    first_name = update.effective_user.first_name
    last_name = update.effective_user.last_name
    avatar = get_avatar(user_data)
    text = "Привет, {} {}! Я бот, который понимает команду /start".format(first_name, avatar)
    logging.info("Пользователь {} {} нажал {}".format(first_name, last_name, "/start"))
    update.message.reply_text(text)


def count_words(bot, update, args, user_data):
    avatar = get_avatar(user_data)
    text = "{} количество слов {}".format(avatar, len(args))
    update.message.reply_text(text)


def chat_with_user(bot, update, user_data):
    avatar = get_avatar(user_data)
    text = "{} {}".format(update.message.text, avatar)
    update.message.reply_text(text)


def send_cat(bot, update, user_data):
    cat_list = glob1("cats", "*.jp*g")
    cat_pic = os.path.join('cats', random.choice(cat_list))
    chat_id = update.message.chat_id
    bot.send_photo(chat_id=chat_id, photo=open(cat_pic, 'rb'))


def start_bot():
    my_bot = Updater(settings.TELEGRAM_API_KEY)

    dp = my_bot.dispatcher
    dp.add_handler(CommandHandler("start", reply_to_start_command, pass_user_data=True))
    dp.add_handler(CommandHandler("countwords", count_words, pass_args=True, pass_user_data=True))
    dp.add_handler(CommandHandler("cat", send_cat, pass_user_data=True))

    dp.add_handler(MessageHandler(Filters.text, chat_with_user, pass_user_data=True))
    my_bot.start_polling()
    my_bot.idle()


if __name__ == "__main__":
    start_bot()
