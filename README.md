# Disaster-Tweets
Code for Kaggle Natural Language Processing with Disaster Tweets Competition
Here's a sample `README.md` for your Disaster Tweets Classification project:

---

# Disaster Tweets Classification

This project is designed to classify tweets as either disaster-related (1) or non-disaster-related (0). The project is based on the Kaggle competition "Real or Not? NLP with Disaster Tweets." It uses various Natural Language Processing (NLP) techniques and machine learning models to predict whether a given tweet is referring to a real disaster.

## Table of Contents
- [Project Overview]
- [Dataset]
- [Requirements]
- [Installation]
- [Preprocessing]
- [Modeling]
- [Sentiment Analysis]
- [Evaluation]
- [Submission]
- [Usage]
- [Contributing]
- [License]

## Project Overview
The goal of this project is to build a machine learning model that can automatically classify tweets as disaster-related or not. The project involves:
- Text preprocessing
- Feature extraction using Bag of Words (BoW) and TF-IDF
- Training machine learning models including Logistic Regression, Naive Bayes, and Support Vector Machines (SVC)
- Sentiment analysis to augment tweet features
- Using soft voting with hyperparameter tuning for optimal model performance

## Dataset
The dataset for this project is sourced from the Kaggle competition: [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data). It consists of a training set and a test set:
- **Train dataset**: Contains labeled tweets with `target` values of 0 (non-disaster) or 1 (disaster).
- **Test dataset**: Contains unlabeled tweets for which predictions are to be made.

## Requirements
- Python 3.6+
- Required libraries are listed in the `requirements.txt` file.

The project utilizes the following key libraries:
- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `textblob`
- `joblib`

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/disaster-tweets-classification.git
    cd disaster-tweets-classification
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # For Windows: env\Scripts\activate
    ```


4. Download the dataset from Kaggle and place the CSV files (`train.csv` and `test.csv`) in the root directory of this project.

## Preprocessing
The text preprocessing steps include:
- Lowercasing the text
- Removing URLs, special characters, and numbers
- Removing stopwords using NLTK
- Stemming using `PorterStemmer`
- Lemmatizing using `WordNetLemmatizer`

Example:
```python
# Preprocessing example
df['cleaned_text'] = df['text'].apply(preprocess_text)
```

## Modeling
The project uses a variety of machine learning models:
- **Logistic Regression**
- **Naive Bayes**
- **Support Vector Machines (SVC)**

We use a **Voting Classifier** to combine these models and improve prediction accuracy. A grid search is performed to optimize hyperparameters for these models.

```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=200)), 
    ('nb', MultinomialNB()), 
    ('svc', SVC(probability=True))
], voting='soft')

random_search = RandomizedSearchCV(estimator=voting_clf, param_distributions=param_grid, 
                                   n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
```

## Sentiment Analysis
We incorporate sentiment analysis using both `VADER` and `TextBlob` to add sentiment scores as features to the model.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

df['sentiment'] = df['text'].apply(lambda text: analyser.polarity_scores(text)['compound'])
```

## Evaluation
The model's performance is evaluated using cross-validation and metrics such as accuracy, precision, recall, and F1-score. The best model is selected using Randomized Search for hyperparameter tuning.

```python
from sklearn.metrics import classification_report

y_pred = random_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Submission
After training the model and making predictions, you can create a CSV file for submission to the Kaggle competition.

```python
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'target': predictions  # Predictions from the best model
})
submission_df.to_csv('disaster_tweets_submission.csv', index=False)
```

## Usage
1. Preprocess the tweets in the dataset using the provided preprocessing functions.
2. Train the model using the training data (`train.csv`).
3. Use the trained model to predict disaster-related tweets from the test set (`test.csv`).
4. Generate a submission file to upload to Kaggle.

To run the pipeline:
```bash
python main.py
```

## Contributing
If you'd like to contribute to this project, please fork the repository and create a pull request with detailed descriptions of your changes. Feel free to submit issues for bugs or feature requests.

## License
This project is licensed under the MIT License
