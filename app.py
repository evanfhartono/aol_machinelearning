import joblib
import re
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime
import json

# Load models
print("LOAD MODEL")
print("word2vec_model...")
w2v_model = joblib.load('models/word2vec_model.pkl')
print("title_ridge_model...")
title_ridge = joblib.load('models/title_ridge_model.pkl')
print("tags_ridge_model...")
tags_ridge = joblib.load('models/tags_ridge_model.pkl')
print("description_ridge_model...")
description_ridge = joblib.load('models/description_ridge_model.pkl')
print("xgb_model_all_genres...")
xgb_model = joblib.load('models/xgb_model_all_genres.pkl')

# Load JSON data
print("Load Channels...")
with open('channels.json', 'r', encoding='utf-8') as f:
    channels = json.load(f)
print("Load categories...")
with open('categories.json', 'r', encoding='utf-8') as f:
    categories = json.load(f)

def custom_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens

def get_avg_vector(tokens, w2v_model):
    vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

def predict_score(text, w2v_model, ridge_model, is_tags=False):
    if is_tags:
        tokens = [tag.strip('"').lower() for tag in text.split('|')]
    else:
        tokens = custom_preprocess(text)
    vector = get_avg_vector(tokens, w2v_model)
    score = ridge_model.predict([vector])[0]
    return score

app = Flask(__name__)

@app.route('/')
def home():
    sorted_channels = sorted(channels.keys())
    sorted_categories = sorted(categories.items(), key=lambda x: x[1])  # Sort by category name
    return render_template('index.html', channels=sorted_channels, categories=sorted_categories)

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title', '')
    tags = request.form.get('tags', '')
    description = request.form.get('description', '')
    published_date_str = request.form.get('published_date', '')
    comments_disabled = 'comments_disabled' in request.form
    ratings_disabled = 'ratings_disabled' in request.form
    category_id_str = request.form.get('categoryId')
    channel_title = request.form.get('channelTitle')

    # Validate inputs
    if not all([title, tags, description, published_date_str, category_id_str, channel_title]):
        sorted_channels = sorted(channels.keys())
        sorted_categories = sorted(categories.items(), key=lambda x: x[1])
        return render_template('index.html', error="Please fill in all fields.", channels=sorted_channels, categories=sorted_categories)

    try:
        published_date = datetime.strptime(published_date_str, '%Y-%m-%d')
        year = published_date.year
        month = published_date.month
        day = published_date.day
    except ValueError:
        sorted_channels = sorted(channels.keys())
        sorted_categories = sorted(categories.items(), key=lambda x: x[1])
        return render_template('index.html', error="Invalid date format. Use YYYY-MM-DD.", channels=sorted_channels, categories=sorted_categories)

    try:
        category_id = int(category_id_str)
    except ValueError:
        sorted_channels = sorted(channels.keys())
        sorted_categories = sorted(categories.items(), key=lambda x: x[1])
        return render_template('index.html', error="Invalid category selected.", channels=sorted_channels, categories=sorted_categories)

    if channel_title not in channels:
        sorted_channels = sorted(channels.keys())
        sorted_categories = sorted(categories.items(), key=lambda x: x[1])
        return render_template('index.html', error="Invalid channel selected.", channels=sorted_channels, categories=sorted_categories)

    # Get encoded value for channelId using channelTitle's encoding
    encoded_channel_id = channels[channel_title]

    # Compute scores
    title_score = predict_score(title, w2v_model, title_ridge)
    tags_score = predict_score(tags, w2v_model, tags_ridge, is_tags=True)
    description_score = predict_score(description, w2v_model, description_ridge)

    tag_list = tags.split('|')
    tag_count = len(tag_list)

    # Prepare features matching the model's expected set
    features = {
        'categoryId': category_id,
        'comments_disabled': int(comments_disabled),
        'ratings_disabled': int(ratings_disabled),
        'year': year,
        'month': month,
        'date': day,
        'channelId': encoded_channel_id,
        'tag_count': tag_count,
        'title_score': title_score,
        'tags_score': tags_score,
        'description_score': description_score
    }

    # Define column order to match training data
    column_order = [
        'categoryId', 'comments_disabled', 'ratings_disabled',
        'year', 'month', 'date', 'channelId', 'tag_count',
        'title_score', 'tags_score', 'description_score'
    ]

    feature_df = pd.DataFrame([features], columns=column_order)

    # Make prediction
    log_views = xgb_model.predict(feature_df)[0]
    predicted_views = np.expm1(log_views)

    sorted_channels = sorted(channels.keys())
    sorted_categories = sorted(categories.items(), key=lambda x: x[1])
    return render_template('index.html', prediction=f"{predicted_views:,.0f}", channels=sorted_channels, categories=sorted_categories)

if __name__ == '__main__':
    app.run(debug=True)