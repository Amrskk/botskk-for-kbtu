import asyncio, os
from aiogram import Bot
from dotenv import load_dotenv

load_dotenv()
from bot import dp  # your file with handlers (bot.py). Change if named differently.

async def main():
    bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

if __name__ == "__main__":
    asyncio.run(main())
