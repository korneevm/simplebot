from glob import glob1
import logging
import random
import os
from emoji import emojize
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, RegexHandler
from telegram import ReplyKeyboardMarkup, KeyboardButton

logging.basicConfig(level=logging.INFO, filename="bot.log", format='%(asctime)s - %(levelname)s - %(message)s')
avatars = [':smiley_cat:', ':smiling_imp:', ':panda_face:', ':dog:']


def get_avatar(user_data):
    if user_data.get('avatar'):
        return user_data.get('avatar')
    else:
        user_data['avatar'] = emojize(random.choice(avatars), use_aliases=True)
        return user_data['avatar']


def get_keyboard():
    contact_button = KeyboardButton('Контактные данные', request_contact=True)
    location_button = KeyboardButton('Геолокация', request_location=True)
    reply_keyboard = [['Прислать котика', 'Сменить аватарку'], [contact_button, location_button]]
    return reply_keyboard


def reply_to_start_command(bot, update, user_data):
    first_name = update.effective_user.first_name
    last_name = update.effective_user.last_name
    avatar = get_avatar(user_data)
    text = "Привет, {} {}! Я бот, который понимает команду /start".format(first_name, avatar)
    logging.info("Пользователь {} {} нажал {}".format(first_name, last_name, "/start"))
    update.message.reply_text(text, reply_markup=ReplyKeyboardMarkup(get_keyboard(), resize_keyboard=True))


def send_cat(bot, update, user_data):
    cat_list = glob1("cats", "*.jp*g")
    cat_pic = os.path.join('cats', random.choice(cat_list))
    chat_id = update.message.chat_id
    bot.send_photo(chat_id=chat_id, photo=open(cat_pic, 'rb'))


def change_avatar_step1(bot, update, user_data):
    reply_keyboard = []
    for index, ava in enumerate(avatars):
        button = "/avatar {} {}".format(index, ava)
        button = emojize(button, use_aliases=True)
        reply_keyboard.append(button)
    text = 'Выбери аватарку {}'.format(get_avatar(user_data))
    update.message.reply_text(text, reply_markup=ReplyKeyboardMarkup([reply_keyboard], resize_keyboard=True))


def change_avatar_step2(bot, update, args, user_data):
    try:
        ava = avatars[int(args[0])]
        user_data['avatar'] = emojize(ava, use_aliases=True)
        update.message.reply_text('Аватарка изменена', reply_markup=ReplyKeyboardMarkup(get_keyboard(), resize_keyboard=True))
    except(IndexError, ValueError):
        update.message.reply_text('Попробуйте еще раз')


def get_contact(bot, update, user_data):
    print(update.message.contact)
    update.message.reply_text('Спасибо {}'.format(get_avatar(user_data)))


def get_location(bot, update, user_data):
    print(update.message.location)
    update.message.reply_text('Спасибо {}'.format(get_avatar(user_data)))


def start_bot():
    my_bot = Updater(TELEGRAM_API_KEY)

    dp = my_bot.dispatcher
    dp.add_handler(CommandHandler("start", reply_to_start_command, pass_user_data=True))
    dp.add_handler(CommandHandler("cat", send_cat, pass_user_data=True))
    dp.add_handler(RegexHandler("^(Прислать котика)$", send_cat, pass_user_data=True))
    dp.add_handler(RegexHandler("^(Сменить аватарку)$", change_avatar_step1, pass_user_data=True))
    dp.add_handler(CommandHandler("avatar", change_avatar_step2, pass_args=True, pass_user_data=True))
    dp.add_handler(MessageHandler(Filters.contact, get_contact, pass_user_data=True))
    dp.add_handler(MessageHandler(Filters.location, get_location, pass_user_data=True))

    my_bot.start_polling()
    my_bot.idle()


if __name__ == "__main__":
    start_bot()
