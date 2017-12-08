import logging
from telegram.ext import Updater, CommandHandler, messagequeue
import settings

logging.basicConfig(level=logging.INFO, filename="bot.log", format='%(asctime)s - %(levelname)s - %(message)s')

chats = []


def reply_to_start_command(bot, update):
    update.message.reply_text("Привет!")


def subscribe_command(bot, update):
    if update.message.chat_id not in chats:
        chats.append(update.message.chat_id)
    update.message.reply_text("Вы подписались на уведомления")


def alarm_command(bot, update, args, job_queue):
    try:
        seconds = abs(int(args[0]))
        job_queue.run_once(alarm, seconds, context=update.message.chat_id)
    except (IndexError, ValueError):
        update.message.reply_text("Введите число секунд после команды /alarm")


def alarm(bot, job):
    bot.sendMessage(chat_id=job.context, text="Сработал будильник!")


@messagequeue.queuedmessage
def my_test(bot, job):
    for chat_id in chats:
        bot.sendMessage(chat_id=chat_id, text="Уведомление")


def start_bot():
    my_bot = Updater(settings.TELEGRAM_API_KEY)

    my_bot.bot._msg_queue = messagequeue.MessageQueue()
    my_bot.bot._is_messages_queued_default = True

    jobs = my_bot.job_queue
    jobs.run_repeating(my_test, interval=5)

    dp = my_bot.dispatcher

    dp.add_handler(CommandHandler("start", reply_to_start_command))
    dp.add_handler(CommandHandler("subscribe", subscribe_command))
    dp.add_handler(CommandHandler("alarm", alarm_command, pass_args=True, pass_job_queue=True))

    my_bot.start_polling()
    my_bot.idle()


if __name__ == "__main__":
    start_bot()
