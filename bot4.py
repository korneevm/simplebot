from glob import glob1
import logging
import random
import os
from emoji import emojize
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, RegexHandler
from telegram import ReplyKeyboardMarkup, KeyboardButton
from db import db_session, User

logging.basicConfig(level=logging.INFO, filename="bot.log", format='%(asctime)s - %(levelname)s - %(message)s')
avatars = [':smiley_cat:', ':smiling_imp:', ':panda_face:', ':dog:']


def get_avatar(user_data):
    if user_data.get('user'):
        return user_data.get('user').avatar
    else:
        return ''


def get_user(effective_user, user_data):
    if not user_data.get('user'):
        user = User.query.filter(User.telegram_id == effective_user.id).first()
        if not user:
            user = User(
                telegram_id=effective_user.id,
                first_name=effective_user.first_name,
                last_name=effective_user.last_name,
                avatar=emojize(random.choice(avatars), use_aliases=True)
            )
            db_session.add(user)
            db_session.commit()
        user_data['user'] = user
        return user
    else:
        return user_data['user']


def get_keyboard():
    contact_button = KeyboardButton('Контактные данные', request_contact=True)
    button2 = KeyboardButton('Геолокация', request_location=True)
    reply_keyboard = [['Прислать котика', 'Сменить аватарку'], [contact_button, button2]]
    return reply_keyboard


def reply_to_start_command(bot, update, user_data):
    user = get_user(update.effective_user, user_data)
    text = "Привет, {} {}! Я бот, который понимает команду /start".format(user.first_name, user.avatar)
    logging.info("Пользователь {} {} нажал {}".format(user.first_name, user.last_name, "/start"))
    update.message.reply_text(text, reply_markup=ReplyKeyboardMarkup(get_keyboard(), resize_keyboard=True))


def send_cat(bot, update, user_data):
    user = get_user(update.effective_user, user_data)
    cat_list = glob1("cats", "*.jp*g")
    cat_pic = os.path.join('cats', random.choice(cat_list))
    chat_id = update.message.chat_id
    bot.send_photo(chat_id=chat_id, photo=open(cat_pic, 'rb'))


def change_avatar_step1(bot, update, user_data):
    user = get_user(update.effective_user, user_data)
    reply_keyboard = []
    for index, ava in enumerate(avatars):
        button = "/avatar {} {}".format(index, ava)
        button = emojize(button, use_aliases=True)
        reply_keyboard.append(button)
    text = 'Выбери аватарку {}'.format(get_avatar(user_data))
    update.message.reply_text(text, reply_markup=ReplyKeyboardMarkup([reply_keyboard], resize_keyboard=True))


def change_avatar_step2(bot, update, args, user_data):
    user = get_user(update.effective_user, user_data)
    try:
        ava = avatars[int(args[0])]
        user.avatar = emojize(ava, use_aliases=True)
        db_session.commit()
        user_data['avatar'] = emojize(ava, use_aliases=True)
        update.message.reply_text('Аватарка изменена', reply_markup=ReplyKeyboardMarkup(get_keyboard(), resize_keyboard=True))
    except (ValueError, IndexError):
        update.message.reply_text('Попробуйте еще раз')


def get_contact(bot, update, user_data):
    user = get_user(update.effective_user, user_data)
    user.phone = update.message.contact.phone_number
    db_session.commit()
    update.message.reply_text('Спасибо {}'.format(get_avatar(user_data)))


def get_location(bot, update, user_data):
    user = get_user(update.effective_user, user_data)
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
