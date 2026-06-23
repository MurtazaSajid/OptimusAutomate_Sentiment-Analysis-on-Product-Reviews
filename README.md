# Sentiment Analysis on Product Reviews

This project is part of the Optimus Automate Machine Learning Internship. The goal is to build a sentiment classifier that reads real customer product reviews and predicts whether the sentiment is positive or negative. Three different machine learning models are trained and compared to find which one performs best on this task.

---

## Dataset

The dataset contains 280 real Amazon Alexa customer reviews collected from Kaggle. Each review comes with a star rating from 1 to 5. Reviews with 1 or 2 stars are labeled as negative and reviews with 4 or 5 stars are labeled as positive. The dataset is balanced with 140 reviews per class so the models are not biased toward one label. The dataset is stored locally as a CSV file named product_reviews.csv which should be placed in the same folder as the script before running.

The CSV file has two columns:

- review: the text of the customer review
- sentiment: the label, either positive or negative

---

## Project Structure

```
Task4_Sentiment/
    sentiment.py          main script
    product_reviews.csv         dataset file
    sentiment_distribution.png  output chart showing class balance
    model_comparison.png        output chart comparing all three models
    confusion_matrix.png        output confusion matrix for best model
```

---

## Steps Performed

### Step 1: Load the Dataset
The dataset is loaded from a local CSV file using pandas. The number of rows, columns, and a sample of the data is printed along with the count per sentiment class.

### Step 2: Clean and Preprocess the Text
Each review goes through a basic cleaning function that converts the text to lowercase and removes punctuation and numbers, keeping only alphabetic characters and spaces. This makes the text consistent and easier for the vectorizer to process.

### Step 3: Visualize Sentiment Distribution
A bar chart is generated showing how many reviews belong to each sentiment class. This confirms the dataset is balanced before training. The chart is saved as sentiment_distribution.png.

### Step 4: Convert Text to TF-IDF Features
The cleaned reviews are converted to numerical features using TF-IDF, which stands for Term Frequency Inverse Document Frequency. This technique gives higher weight to words that appear often in one review but not across all reviews, making it better at capturing meaningful words. Stopwords like "the", "is", and "a" are automatically removed during this step. The vocabulary is capped at 1000 features and words that appear in fewer than 2 reviews are dropped to reduce noise.

### Step 5: Split Data into Train and Test Sets
The data is split into 80 percent training and 20 percent testing using a stratified split to keep the class ratio equal in both sets.

### Step 6: Train the Models
Three models are trained and compared.

Naive Bayes is a simple probabilistic model that works well for text classification and is fast to train. SVM (Support Vector Machine) uses a linear kernel to find the best boundary between positive and negative reviews. Class weights are set to balanced so neither class is treated as more important. A small Neural Network with a single hidden layer of 30 neurons is also trained. Using only one hidden layer and a regularization parameter keeps it from memorizing the training data.

### Step 7: Evaluate the Models
All three models are evaluated using accuracy, precision, recall, and F1-score. Results for each model are printed in the console.

### Step 8: Visualize Model Performance
A grouped bar chart comparing all four metrics across all three models is saved as model_comparison.png. This gives a clear picture of which model performs best and in which areas.

### Step 9: Confusion Matrix for the Best Model
The model with the highest F1-score is selected and its confusion matrix is generated and saved as confusion_matrix.png. The confusion matrix shows exactly how many reviews were correctly classified and where the model made mistakes.

### Step 10: Summary
The best model name and its accuracy are printed at the end of the run, along with the paths where all output files were saved.

---

## How to Run

1. Make sure Python is installed on your machine.
2. Install the required libraries by running this command in your terminal or PyCharm terminal:

```
pip install pandas scikit-learn matplotlib seaborn
```

3. Place sentiment_task4.py and product_reviews.csv in the same folder.
4. Run the script:

```
python sentiment.py
```

5. The three output charts will be saved automatically in the same folder as the script.

---

## Results

SVM achieved the highest accuracy at around 89 percent on the test set. Naive Bayes and the Neural Network both reached approximately 87 to 88 percent. The high accuracy comes from the fact that positive and negative Amazon reviews tend to use very distinct vocabulary, which TF-IDF is good at capturing. Words like "love", "amazing", and "perfect" strongly signal positive reviews while words like "broke", "disappointed", and "waste" strongly signal negative ones.

---

## Libraries Used

- pandas
- scikit-learn
- matplotlib
- seaborn
