import os
import cv2
import numpy as np
import logging
import asyncio
import aiofiles
from io import BytesIO
from PIL import Image
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ConversationHandler, filters, ContextTypes
)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Состояния для ConversationHandler
PHOTO_SRC, PHOTO_DST = range(2)

# Директории для временного хранения
TEMP_DIR = "temp_deepfake"
os.makedirs(TEMP_DIR, exist_ok=True)

# Глобальная переменная для модели (загружается один раз при старте)
swapper_model = None

# ================== ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ==================
def load_models():
    """Загрузка модели замены лиц InsightFace"""
    global swapper_model
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    
    logger.info("Загрузка моделей InsightFace...")
    
    # Инициализация детектора лиц (RetinaFace)
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Загрузка модели сваппера (inswapper_128.onnx)
    swapper = get_model('inswapper_128.onnx', download=True, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    logger.info("Модели успешно загружены")
    return app, swapper

# ================== ФУНКЦИИ ОБРАБОТКИ ИЗОБРАЖЕНИЙ ==================
async def swap_faces(src_img_path: str, dst_img_path: str, output_path: str) -> bool:
    """
    Замена лица с фото src_img_path на лицо на фото dst_img_path.
    Возвращает True при успехе.
    """
    try:
        # Запускаем в отдельном потоке, чтобы не блокировать event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _swap_faces_sync, src_img_path, dst_img_path, output_path)
        return result
    except Exception as e:
        logger.error(f"Ошибка в swap_faces: {e}", exc_info=True)
        return False

def _swap_faces_sync(src_img_path: str, dst_img_path: str, output_path: str):
    """Синхронная функция замены лиц (выполняется в thread pool)"""
    global swapper_model
    
    if swapper_model is None:
        app, swapper = load_models()
    else:
        app, swapper = swapper_model
    
    # Чтение изображений
    src_img = cv2.imread(src_img_path)
    dst_img = cv2.imread(dst_img_path)
    
    if src_img is None or dst_img is None:
        raise ValueError("Не удалось прочитать изображения")
    
    # Детекция лиц на исходном фото (донор)
    src_faces = app.get(src_img)
    if len(src_faces) == 0:
        raise ValueError("На исходном фото не найдено лиц")
    
    # Берем лицо с наибольшим размером (основное лицо)
    src_face = sorted(src_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    
    # Детекция лиц на целевом фото
    dst_faces = app.get(dst_img)
    if len(dst_faces) == 0:
        raise ValueError("На целевом фото не найдено лиц")
    
    # Замена лиц
    result_img = dst_img.copy()
    for face in dst_faces:
        # Применяем сваппер для каждого лица на целевом фото
        result_img = swapper.get(result_img, face, src_face, paste_back=True)
    
    # Сохранение результата
    cv2.imwrite(output_path, result_img)
    return True

# ================== ХЭНДЛЕРЫ TELEGRAM ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Команда /start - начало работы"""
    await update.message.reply_text(
        "👋 Привет! Я бот для замены лиц (дипфейк).\n\n"
        "📸 Отправь мне фото человека, **чьё лицо** нужно вставить (донор)."
    )
    return PHOTO_SRC

async def photo_src(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получение фото-донора"""
    user = update.effective_user
    photo_file = await update.message.photo[-1].get_file()
    
    # Создаем уникальные имена файлов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_path = os.path.join(TEMP_DIR, f"{user.id}_{timestamp}_src.jpg")
    
    # Скачиваем файл асинхронно
    await photo_file.download_to_drive(src_path)
    context.user_data['src_path'] = src_path
    
    await update.message.reply_text(
        "✅ Фото донора сохранено.\n"
        "📸 Теперь отправь фото, **на котором нужно заменить лицо** (цель)."
    )
    return PHOTO_DST

async def photo_dst(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получение фото-цели и запуск обработки"""
    user = update.effective_user
    photo_file = await update.message.photo[-1].get_file()
    
    # Проверяем наличие фото донора
    if 'src_path' not in context.user_data:
        await update.message.reply_text("❌ Сначала отправь фото донора. Начни заново с /start")
        return ConversationHandler.END
    
    # Сохраняем фото цели
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_path = os.path.join(TEMP_DIR, f"{user.id}_{timestamp}_dst.jpg")
    output_path = os.path.join(TEMP_DIR, f"{user.id}_{timestamp}_result.jpg")
    
    await photo_file.download_to_drive(dst_path)
    
    # Отправляем сообщение о начале обработки
    status_msg = await update.message.reply_text("⏳ Обрабатываю изображения... Это может занять до 30 секунд.")
    
    try:
        # Выполняем замену лиц
        success = await swap_faces(context.user_data['src_path'], dst_path, output_path)
        
        if success:
            # Отправляем результат
            with open(output_path, 'rb') as f:
                await update.message.reply_photo(
                    photo=f,
                    caption="✅ Готово! Лицо заменено."
                )
            await status_msg.delete()
        else:
            await status_msg.edit_text("❌ Не удалось обработать изображения. Проверьте, что на фото есть лица.")
    
    except ValueError as e:
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}", exc_info=True)
        await status_msg.edit_text("❌ Произошла внутренняя ошибка.")
    
    finally:
        # Очистка временных файлов
        try:
            os.remove(context.user_data['src_path'])
            os.remove(dst_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass
        
        # Очищаем данные пользователя
        context.user_data.clear()
    
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена операции"""
    await update.message.reply_text("🚫 Операция отменена. Начни заново с /start")
    
    # Очистка временных файлов
    if 'src_path' in context.user_data:
        try:
            os.remove(context.user_data['src_path'])
        except:
            pass
    
    context.user_data.clear()
    return ConversationHandler.END

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    help_text = """
🤖 **DeepFake Swap Bot**

**Как пользоваться:**
1. Отправь `/start`
2. Загрузи фото человека, чье лицо нужно вставить (донор)
3. Загрузи фото, на котором нужно заменить лицо (цель)
4. Подожди 10-30 секунд, бот пришлет результат

**Важно:**
- На фото должно быть четко видно лицо
- Бот не хранит ваши фотографии (удаляются сразу после обработки)
- Только для развлекательных целей

**Команды:**
/start - начать работу
/help - эта справка
/cancel - отменить текущую операцию
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок"""
    logger.error(f"Ошибка: {context.error}", exc_info=context.error)
    if update and update.effective_message:
        await update.effective_message.reply_text("⚠️ Произошла техническая ошибка. Попробуйте позже.")

# ================== ЗАПУСК БОТА ==================
def main():
    """Основная функция запуска"""
    # Токен бота (замените на свой)
    TOKEN = "8366196461:AAHD95QhRtLPd9sPdJFP5X9wZun_FVEx4Ww"
    
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()
    
    # Загружаем модели при старте (в фоне, чтобы не задерживать запуск)
    async def preload_models(app):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_models)
    
    application.post_init = preload_models
    
    # Обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            PHOTO_SRC: [MessageHandler(filters.PHOTO, photo_src)],
            PHOTO_DST: [MessageHandler(filters.PHOTO, photo_dst)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    
    # Регистрируем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('help', help_command))
    
    # Обработчик ошибок
    application.add_error_handler(error_handler)
    
    # Запуск бота
    print("🤖 Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
