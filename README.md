# SMS Spam Detector

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP) and the Naive Bayes algorithm.

## Overview

This project builds a spam detection model using the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). It processes raw text messages with TF-IDF vectorization and trains a Multinomial Naive Bayes classifier to identify spam messages.

## Features

- Text preprocessing with TF-IDF vectorization
- Spam classification using Multinomial Naive Bayes
- Model evaluation with accuracy, confusion matrix, and classification report
- Saves trained model and vectorizer for reuse
- Interactive testing loop to classify custom messages

## Tech Stack

- **Python**
- **pandas** — data loading and manipulation
- **scikit-learn** — TF-IDF vectorization, model training, and evaluation
- **joblib** — model serialization

## Dataset

The project uses `spam.csv` (SMS Spam Collection Dataset) containing 5,574 SMS messages labeled as `ham` or `spam`.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/spam-detector.git
   cd spam-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```bash
python main.py
```

The script will:

1. Load and preprocess the dataset
2. Split data into training (80%) and testing (20%) sets
3. Vectorize text using TF-IDF
4. Train a Multinomial Naive Bayes classifier
5. Evaluate the model and print metrics
6. Save the trained model (`spam_model.pkl`) and vectorizer (`vectorizer.pkl`)
7. Start an interactive loop where you can test your own messages

### Interactive Testing

After training, you can type any message to check if it's spam:

```
Enter a message: Congratulations! You've won a free ticket. Call now!
🚨 SPAM MESSAGE

Enter a message: Hey, are we still meeting for lunch tomorrow?
✅ NOT SPAM

Enter a message: exit
Exiting...
```

## Project Structure

```
spam-detector/
├── main.py             # Training, evaluation, and interactive testing
├── spam.csv            # SMS Spam Collection dataset
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## License

This project is open source and available under the [MIT License](LICENSE).