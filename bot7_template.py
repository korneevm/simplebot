import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, RegexHandler, ConversationHandler
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove

logging.basicConfig(level=logging.INFO, filename="bot.log", format='%(asctime)s - %(levelname)s - %(message)s')


def reply_to_start_command(bot, update):
    reply_keyboard = [['Прислать котика', 'Узнать погоду', 'Заполнить анкету']]
    update.message.reply_text(
        "Привет! Я бот, который поможет заполнить простую анекту.",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard)
    )


def start_bot():
    my_bot = Updater('TELEGRAM_API_KEY')

    dp = my_bot.dispatcher

    dp.add_handler(CommandHandler("start", reply_to_start_command))

    my_bot.start_polling()
    my_bot.idle()


if __name__ == "__main__":
    start_bot()
