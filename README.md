# VeriScan - AI-Powered Fake News Detection & Fact Verification

## Overview

VeriScan is an AI-powered fake news detection system that combines Machine Learning, Google Fact Check API, and Wikipedia verification to evaluate the credibility of news articles. The system uses a multi-layer verification pipeline to provide reliable and explainable fact-checking results.

## Features

* Fake news detection using Machine Learning
* 4-layer credibility verification pipeline
* Google Fact Check API integration
* Wikipedia-based semantic verification
* Multi-source fact validation
* Real-time news credibility assessment
* User-friendly Flask web interface

## Tech Stack

* Python
* Flask
* Scikit-learn
* NLTK
* Google Fact Check API
* Wikipedia API
* HTML, CSS, JavaScript

## Machine Learning Model

### Dataset

* **WELFake Dataset**
* Total Articles: **72,134**

### Model Performance

| Vectorization Method | Accuracy |
| -------------------- | -------- |
| Bag of Words (BoW)   | 88.09%   |
| TF-IDF               | 85.65%   |
| Word2Vec             | 79.24%   |

### Final Model

* Multinomial Naïve Bayes
* 5,000-feature Bigram Bag-of-Words Vectorizer
* Accuracy: **88.09%**
* F1 Score: **0.88**

## Verification Pipeline

### Layer 1: Machine Learning Classification

Predicts whether the news article is likely real or fake using the trained Multinomial Naïve Bayes model.

### Layer 2: Google Fact Check Verification

Searches trusted fact-checking sources such as:

* PolitiFact
* Snopes
* AFP Fact Check
* Other Google Fact Check partners

### Layer 3: Rating Analysis

Uses a keyword-based rating classifier with 15+ credibility indicators to analyze fact-check verdicts.

### Layer 4: Wikipedia Verification

Performs semantic validation by comparing article claims with relevant Wikipedia knowledge sources.

## Project Structure

```text
VeriScan/
│
├── app.py
├── model/
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   └── js/
│
├── dataset/
│   └── WELFake_Dataset.csv
│
├── requirements.txt
└── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/Saanchi0505/veriscan-fake-news-detection.git
cd veriscan-fake-news-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open your browser and visit:

```text
http://localhost:5000
```

## Results

* Trained on 72,134 news articles.
* Evaluated on 17,871 holdout samples.
* Achieved 88.09% accuracy and 0.88 F1-score.
* Outperformed TF-IDF and Word2Vec approaches.

## Future Enhancements

* Deep Learning-based models (LSTM, GRU, BERT)
* News source credibility scoring
* Explainable AI visualizations
* Browser extension integration
* Multilingual fake news detection

## Author

**Saanchi**

Developed independently as an end-to-end Machine Learning and Fact Verification project.
