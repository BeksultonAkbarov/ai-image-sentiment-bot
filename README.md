AI Image Sentiment Analysis Telegram Bot

A simple Telegram bot that classifies uploaded images as positive or negative sentiment using a PyTorch CNN trained on the EmotionROI dataset. Built for an AI course term project.
Features

Receives photos via Telegram.
Predicts sentiment: "Positive" or "Negative".
Trained on ~200 images (balanced positive/negative emotions).
Local run for demo; accuracy ~85%.

Requirements

Python 3.10+
Libraries: torch, torchvision, telebot, pillow, scikit-learn, python-dotenv

Install:
Bashpip install torch torchvision torchaudio telebot pillow scikit-learn python-dotenv
Setup

Dataset: Download EmotionROI ZIP. Unzip to data/ with subfolders positive/ and negative/. Balance ~100 images each.
Bot Token: Create bot via @BotFather. Add to .env:textBOT_TOKEN=your_token_hereAdd .env to .gitignore.
Train Model:Bashpython train.py
Runs 30 epochs with augmentation/dropout.
Saves image_sentiment.pth.

Run Bot:Bashpython bot.py
Starts polling. Send photos to bot in Telegram.
Replies with prediction.


Code Structure

train.py: Loads data, trains CNN, saves model.
bot.py: Loads model, handles photo uploads, predicts.
CNN: Simple conv layers + dropout for classification.
