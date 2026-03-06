import os
import cv2
import numpy as np
import logging
import asyncio
from io import BytesIO
from PIL import Image
import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Улучшенные модели
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import facexlib
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighQualitySwapper:
    """
    Класс для высококачественной замены лиц с пост-обработкой
    """
    
    def __init__(self):
        logger.info("Загрузка улучшенных моделей...")
        
        # 1. Детектор лиц (RetinaFace с высоким разрешением)
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 2. Основная модель замены (inswapper)
        self.swapper = get_model(
            'inswapper_128.onnx',
            download=True,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 3. GFPGAN для восстановления деталей лица
        self.gfpgan = GFPGANer(
            model_path='GFPGANv1.4.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        
        # 4. Real-ESRGAN для апскейлинга
        self.esrgan = RealESRGANer(
            scale=2,
            model_path='RealESRGAN_x2plus.pth',
            tile=400,
            tile_pad=10,
            pre_pad=0
        )
        
        logger.info("Все модели загружены")
    
    def swap_and_enhance(self, src_img: np.ndarray, dst_img: np.ndarray) -> np.ndarray:
        """
        Полный конвейер: замена лица + улучшение качества
        """
        # Детекция лиц на исходном фото
        src_faces = self.app.get(src_img)
        if len(src_faces) == 0:
            raise ValueError("На исходном фото не найдено лиц")
        
        # Выбираем самое крупное лицо
        src_face = max(src_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        
        # Детекция лиц на целевом фото
        dst_faces = self.app.get(dst_img)
        if len(dst_faces) == 0:
            raise ValueError("На целевом фото не найдено лиц")
        
        # Замена лиц
        result = dst_img.copy()
        for face in dst_faces:
            result = self.swapper.get(result, face, src_face, paste_back=True)
        
        # Улучшение через GFPGAN (восстановление текстур)
        result, _ = self.gfpgan.enhance(
            result,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        
        # Апскейлинг в 2 раза (опционально)
        if max(result.shape[:2]) < 1024:
            result, _ = self.esrgan.enhance(result, outscale=2)
        
        return result

# Telegram бот
TOKEN = "8366196461:AAHD95QhRtLPd9sPdJFP5X9wZun_FVEx4Ww"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Отправь два фото:\n"
        "1. Фото человека, чье лицо нужно вставить\n"
        "2. Фото, на котором заменить лицо\n\n"
        "Качество будет максимальным!"
    )

async def handle_photos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    # Сохраняем фото
    if 'src_photo' not in context.user_data:
        photo_file = await update.message.photo[-1].get_file()
        src_path = f"temp/{user.id}_src.jpg"
        await photo_file.download_to_drive(src_path)
        context.user_data['src_photo'] = src_path
        await update.message.reply_text("✅ Фото донора сохранено. Теперь отправьте целевое фото.")
        return
    
    # Получаем целевое фото
    photo_file = await update.message.photo[-1].get_file()
    dst_path = f"temp/{user.id}_dst.jpg"
    await photo_file.download_to_drive(dst_path)
    
    # Загружаем исходное
    src_path = context.user_data['src_photo']
    
    await update.message.reply_text("⏳ Обработка (улучшенное качество)...")
    
    try:
        # Читаем изображения
        src_img = cv2.imread(src_path)
        dst_img = cv2.imread(dst_path)
        
        # Инициализируем сваппер (ленивая загрузка)
        if 'swapper' not in context.bot_data:
            context.bot_data['swapper'] = HighQualitySwapper()
        
        swapper = context.bot_data['swapper']
        
        # Замена с улучшением
        result = swapper.swap_and_enhance(src_img, dst_img)
        
        # Сохраняем результат
        output_path = f"temp/{user.id}_result.jpg"
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        # Отправляем
        with open(output_path, 'rb') as f:
            await update.message.reply_photo(
                photo=f,
                caption="✅ Готово! Улучшенное качество"
            )
        
        # Очистка
        os.remove(src_path)
        os.remove(dst_path)
        os.remove(output_path)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
    
    context.user_data.clear()

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photos))
    app.run_polling()

if __name__ == "__main__":
    main()
