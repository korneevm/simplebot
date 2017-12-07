import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, RegexHandler, ConversationHandler
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
import settings

logging.basicConfig(level=logging.INFO, filename="bot.log", format='%(asctime)s - %(levelname)s - %(message)s')


def reply_to_start_command(bot, update):
    reply_keyboard = [['Прислать котика', 'Узнать погоду', 'Заполнить анкету']]
    update.message.reply_text(
        "Привет! Я бот, который поможет заполнить простую анекту.",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard)
    )


def start_anketa(bot, update, user_data):
    update.message.reply_text("Напишите, как вас зовут.", reply_markup=ReplyKeyboardRemove())
    return "name"


def get_name(bot, update, user_data):
    user_name = update.message.text
    if len(user_name.split(" ")) < 2:
        update.message.reply_text("Пожалуйста, напишите имя и фамилию")
        return "name"
    else:
        user_data["name"] = user_name
        reply_keyboard = [["1", "2", "3", "4", "5"]]

        update.message.reply_text(
            "Понравился ли вам курс? Оцените по шкале от 1 до 5",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return "attitude"


def attitude(bot, update, user_data):
    user_data["attitude"] = update.message.text

    reply_keyboard = [["1", "2", "3", "4", "5"]]

    update.message.reply_text(
        "Все ли было понтяно? Оцените по шкале от 1 до 5",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    )
    return "understanding"


def understanding(bot, update, user_data):
    user_data["understanding"] = update.message.text
    update.message.reply_text("Оставьте комментарий в свободной форме или пропустите этот шаг, введя /cancel")
    return "comment"


def comment(bot, update, user_data):
    user_data["comment"] = update.message.text
    reply_keyboard = [['Прислать котика', 'Узнать погоду']]
    update.message.reply_text("Спасибо за ваш комментарий!", reply_markup=ReplyKeyboardMarkup(reply_keyboard))
    return ConversationHandler.END


def skip_comment(bot, update, user_data):
    user_data["comment"] = "Пользователь не оставил комментария"
    reply_keyboard = [['Прислать котика', 'Узнать погоду']]
    update.message.reply_text("Спасибо!", reply_markup=ReplyKeyboardMarkup(reply_keyboard))
    return ConversationHandler.END


def dontknow(bot, update, user_data):
    update.message.reply_text("Не понимаю")


def false_start(bot, update, user_data):
    update.message.reply_text("Нельзя :)")


def start_bot():
    my_bot = Updater(settings.TELEGRAM_API_KEY)

    dp = my_bot.dispatcher

    dp.add_handler(CommandHandler("start", reply_to_start_command))

    conv_handler = ConversationHandler(
        entry_points=[RegexHandler('^(Заполнить анкету)$', start_anketa, pass_user_data=True)],

        states={
            "name": [MessageHandler(Filters.text, get_name, pass_user_data=True)],
            "attitude": [RegexHandler('^(1|2|3|4|5)$', attitude, pass_user_data=True)],
            "understanding": [RegexHandler('^(1|2|3|4|5)$', understanding, pass_user_data=True)],
            "comment": [MessageHandler(Filters.text, comment, pass_user_data=True),
                        CommandHandler('skip', skip_comment, pass_user_data=True)],
        },

        fallbacks=[MessageHandler(Filters.text, dontknow, pass_user_data=True)]
    )

    dp.add_handler(conv_handler)

    my_bot.start_polling()
    my_bot.idle()


if __name__ == "__main__":
    start_bot()
