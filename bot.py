import os
import json
import asyncio
from fastapi import FastAPI, Request
from pathlib import Path
import glob
import re

from aiogram.utils.media_group import MediaGroupBuilder
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile

from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
DB_FILE = "chat_history.json"   # optional, read-only curated Q/A
PDF_ROOT = Path(__file__).resolve().parent / "files" / "rups"
YEAR_RANGES = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]

# model selection (switchable via inline buttons)
current_model = {"name": DEFAULT_MODEL}

#FastAPI
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
app = FastAPI()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return json.loads(txt) if txt else []
    except json.JSONDecodeError:
        return []

def search_db(question: str, threshold: float = 0.75):
    db = load_db()
    if not db:
        return None
    questions = [row["question"] for row in db]
    em_db = embedder.encode(questions, convert_to_tensor=True)
    em_q = embedder.encode(question, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(em_q, em_db)[0]
    best_score = float(sims.max())
    best_idx = int(sims.argmax())
    if best_score >= threshold:
        return db[best_idx]["answer"]
    return None

#Initial Prompts
SYSTEM_PROMPT = (
    "Ты дружелюбный Telegram-бот по имени сын Amrskk. Вы очень умный помощник для студентов КБТУ."
    "Отвечай чётко, полезно и кратко. Не упоминай провайдеры API. "
    "Соблюдай нейтралитет и избегай оскорблений."
)

#Inline Keyboards
def inline_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=" Настройкии", callback_data="settings"),
            InlineKeyboardButton(text=" Помощь Хелп", callback_data="help"),
            InlineKeyboardButton(text=" РУПЫ", callback_data="rups"),
            InlineKeyboardButton(text=" Канал Amrskk", url="https://t.me/amrskkyvlposts"),
            InlineKeyboardButton(text=" Академ Календарь", callback_data="calendar")
        ],
        [
            InlineKeyboardButton(text=" Модель: GPT-5", callback_data="model_gpt5"),
            InlineKeyboardButton(text=" Модель: GPT-5-mini", callback_data="model_gpt5mini"),
            InlineKeyboardButton(text=" Формирование Расписания", callback_data="timetable-alter"),
            InlineKeyboardButton(text=" Папка с каналами и чатами КБТУ", callback_data="kbtu-chats")
        ],
    ])

def rups_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="ШИТиИ", callback_data="years:site"),
            InlineKeyboardButton(text="ШМиЗТ", callback_data="years:smsgt"),
            InlineKeyboardButton(text="ШЭиНГИ", callback_data="years:petro"),
            InlineKeyboardButton(text="Кма(багает)", callback_data="years:sea"),
        ],
        [
            InlineKeyboardButton(text="ШХИ", callback_data="years:chem"),
            InlineKeyboardButton(text="ШПМ", callback_data="years:math"),
            InlineKeyboardButton(text="МШЭ", callback_data="years:eco"),
            InlineKeyboardButton(text="БШ",  callback_data="years:bus"),
            InlineKeyboardButton(text="ШГ",  callback_data="years:geo"),
        ],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="menu_root")]
    ])

def years_menu(faculty_slug: str) -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=yr, callback_data=f"{faculty_slug}:{yr}") for yr in YEAR_RANGES]
    return InlineKeyboardMarkup(inline_keyboard=[row, [
        InlineKeyboardButton(text="⬅️ Назад", callback_data="rups")
    ]])


# Commands
@dp.message(F.text.in_({"/start", "/menu"}))
async def cmd_start(msg: Message):
    await msg.answer(
        "Привет! Я сын Amrskk. Welcome to KBTU telegram brochacho"
        "⣿⣿⣿⣿⣿⠅⠈⠉⠈⠀⠀⠀⠁⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿"
        "⣿⣿⣿⣿⡏⠀⡐⠋⢁⡀⢰⣤⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿"
        "⣿⣿⣿⡟⠀⢐⣑⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿"
        "⣿⣿⣿⢏⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠛⠛⠛⢿⣿⣿⡿⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠛⠛⢻⡟⠁⠀⠀⠀⠀⠨⣿⡏⠀⠀⠀⠀⠉⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⣀⣸⠄⠀⠀⠀⠀⠀⠀⢹⣿⠆⠀⠀⠀⠙⣿⡿⠿⣿⣿⣿"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⠈⠉⡄⠀⠀⠀⠀⠀⢸⡄⠀⠀⠀⠀⠐⣿⠀⠀⠀⠀⠀⠀⠀⢸⠁⠀⠀⠀⠀⠀⠋⠀⠀⠀⢻⣿"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⣧⠀⠀⠀⠀⢀⣿⣧⠀⠀⠀⣴⣿⣿⣦⡀⠀⠀⠀⠀⠀⣼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢻"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⡀⠀⠀⠀⣸⣿⣿⣦⠀⣼⣿⣿⣿⣿⣿⣿⣿⣶⣶⣶⣿⣶⣶⣤⣤⣤⡀⠀⠀⠀⠀⠀⣿"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⣼⣿"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⢰⣿⣿"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢠⣿⣿⣿"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⣿⣿⣿⣿"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠗"
        "⡋⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠨"
        "⡀⠀⠸⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠽⠀"
        "⣿⡇⠀⠉⢻⣿⣿⣿⣿⠿⠋⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⠀⢸"
        "⣿⣷⠀⠀⠀⠻⣿⣿⡏⠀⠀⠈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠋⠀⠸⣿⣿⠇⠀⠀⠀⡟"
        "⠈⠁⠀⠀⠀⠀⠉⠙⢻⣦⣀⢀⣸⠟⠁⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠆⠀⠀⠀⢀⣿⠛⠁⠀⠀⠀⠀"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠻⢿⣇⡀⠀⠀⢹⡿⠛⠉⠻⣿⣿⣿⠿⣿⣿⣿⣿⣿⣿⠍⠀⠀⠀⠀⢀⣽⠈⠀⠀⠀⠀⠀⠀"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠛⠓⠶⣾⡆⠀⠀⠀⠈⠉⠀⠀⠈⡟⠁⠀⠉⠀⠀⠀⠀⠀⣀⡜⠉⠀⠀⠀⠀⠀⠀⠠"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠠⠀⠀⠀⠀⠀⠀⠀⠅⠀⠀⠀⢀⣀⣀⡤⠿⠃⠀⠀⠀⠀⠀⠀⠀⢀⣶"
        "⠄⠀⠀⠀⠀⠀⠀⠀⠐⠀⢠⣤⣏⣰⡂⠀⠠⠀⡀⠀⠘⠀⠁⠒⠶⠶⠷⣤⣶⢶⠟⣻⠉⢄⠀⡀⠀⠀⠀⠀⠀⠀⠀⣼⣿",
        reply_markup=inline_main_menu()
    )

async def send_pdfs_as_albums(cb: CallbackQuery, paths: list[str], chunk_size: int = 10):
    if not paths:
        await cb.answer("PDF не найдены", show_alert=True)
        return

    await cb.answer()
    await cb.message.answer(f"Нашёл {len(paths)} файлов, отправляю…")

    for i in range(0, len(paths), chunk_size):
        chunk = paths[i:i+chunk_size]
        mg = MediaGroupBuilder()
        for p in chunk:
            mg.add_document(media=FSInputFile(p), caption=Path(p).name[:100])
        await cb.message.answer_media_group(mg.build())


@dp.callback_query(
    F.data.regexp(r"^(site|smsgt|petro|chem|math|eco|bus|sea|geo):(\d{4}-\d{4})$")
)
async def on_faculty_year(cb: CallbackQuery):
    m = re.match(r"^(site|smsgt|petro|chem|math|eco|bus|sea|geo):(\d{4}-\d{4})$", cb.data)
    faculty, year_range = m.group(1), m.group(2)

    folder = PDF_ROOT / faculty / year_range
    paths = sorted(glob.glob(str(folder / "*.pdf")))
    await send_pdfs_as_albums(cb, paths, chunk_size=10)

#Callback handlers (inline buttons)
@dp.callback_query(F.data == "help")
async def on_help(cb: CallbackQuery):
    text = (
        "Помощь:\n"
        "• Я отвечаю на вопросы и могу переключать LLM модель.\n"
        "• База знаний (если есть) только для чтения — пользователи не могут её менять.\n"
        "• Кнопки ниже для настроек и выбора модели."
    )
    await cb.message.edit_text(text, reply_markup=inline_main_menu())
    await cb.answer()

@dp.callback_query(F.data == "menu_root")
async def on_menu_root(cb: CallbackQuery):
    await cb.message.edit_text("Главное меню:", reply_markup=inline_main_menu())
    await cb.answer()

@dp.callback_query(F.data == "calendar")
async def on_calendar(cb: CallbackQuery):
    p = Path(__file__).resolve().parent / "files" / "Calendar20252026.pdf"
    if not p.exists():
        await cb.answer("Файл не найден на сервере", show_alert=True)
        return
    doc = FSInputFile(str(p), filename="Calendar20252026.pdf")
    await cb.message.answer_document(doc, caption="Академический календарь")
    await cb.answer()


@dp.callback_query(F.data == "settings")
async def on_settings(cb: CallbackQuery):
    text = f"Текущая модель: {current_model['name']}"
    await cb.message.edit_text(text, reply_markup=inline_main_menu())
    await cb.answer()

@dp.callback_query(F.data == "rups")
async def on_rups(cb: CallbackQuery):
    await cb.message.edit_text(
        "Выберите факультет",
        reply_markup=rups_menu()
    )
    await cb.answer()

@dp.callback_query(F.data.regexp(r"^years:(site|smsgt|petro|chem|math|eco|bus|sea|geo)$"))
async def on_years_menu(cb: CallbackQuery):
    faculty = cb.data.split(":", 1)[1]
    await cb.message.edit_text("Выберите учебный год:", reply_markup=years_menu(faculty))
    await cb.answer()


@dp.callback_query(F.data == "kbtu-chats")
async def on_kbtu_chats(cb: CallbackQuery):
    text = (
        "Папка с чатами и каналами КБТУ:\n"
        "https://t.me/addlist/CcIzd-EfozY4MDky"
    )
    await cb.message.edit_text(text, reply_markup=inline_main_menu())
    await cb.answer()

@dp.callback_query(F.data == "timetable-alter")
async def on_timetable_alter(cb: CallbackQuery):
    p = Path(__file__).resolve().parent / "files" / "guideToSchedule.pdf"
    if not p.exists():
        await cb.answer("Файл не найден на сервере", show_alert=True)
        return
    doc = FSInputFile(str(p), filename="guideToSchedule.pdf")
    await cb.message.answer_document(doc, caption="Вот расписание")
    await cb.answer()

@dp.callback_query(F.data.in_({"model_gpt5", "model_gpt5mini"}))
async def on_model_change(cb: CallbackQuery):
    new_name = "gpt-5" if cb.data == "model_gpt5" else "gpt-5-mini"
    current_model["name"] = new_name
    await cb.message.edit_text(f"Модель переключена на: {new_name}", reply_markup=inline_main_menu())
    await cb.answer("Switched")

# Main text handler
@dp.message(F.text)
async def handle_msg(msg: Message):
    user_input = (msg.text or "").strip()
    if not user_input:
        return

    # In groups, only respond when bot is mentioned
    if msg.chat.type in ("group", "supergroup"):
        bot_info = await bot.get_me()
        mentioned = any(
            f"@{bot_info.username}" in (msg.text or "")
            for entity in (msg.entities or [])
            if entity.type in ("mention", "text_mention", "bot_command")
        )
        if not mentioned:
            return

    # Try curated read-only answers first
    cached = search_db(user_input)
    if cached:
        await msg.answer(cached)
        return

    # Build messages and call selected model
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]

    try:
        llm = ChatOpenAI(model=current_model["name"])
        result = await llm.ainvoke(messages)
        reply = result.content
        await msg.answer(reply)
    except Exception as e:
        await msg.answer(f"Ошибка LLM: {e}")

# Webhook endpoint (FastAPI) 
@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    update = types.Update(**data)
    await dp.feed_update(bot, update)
    return {"ok": True}
