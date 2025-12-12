import telebot
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import requests
import time
import os
from dotenv import load_dotenv

MODEL_PATH = 'image_sentiment.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(num_classes=2):
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

load_dotenv()
bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))
model = build_model()
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state['model_state'])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@bot.message_handler(content_types=['photo'])
def predict(msg):
    for attempt in range(3):
        try:
            file_info = bot.get_file(msg.photo[-1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            break
        except requests.exceptions.ConnectionError:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                bot.reply_to(msg, "Download failed—try again.")
                return

    img = Image.open(BytesIO(downloaded_file)).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
        pred = torch.argmax(out, 1).item()

    sentiment = 'Positive' if pred == 1 else 'Negative'
    bot.reply_to(msg, f"Image Sentiment: {sentiment}")


if __name__ == '__main__':
    print('Bot running...')
    bot.polling()

