# Sentiment Analysis of Amazon Apparel Reviews

## Overview

This project focuses on sentiment analysis of customer reviews from the Amazon Apparel category. The goal is to analyse the sentiment (positive, neutral, or negative) expressed in the reviews and evaluate the performance of various machine learning algorithms in predicting sentiment based on the review text. The project was conducted as part of my Master's degree in Big Data Analytics at Sheffield Hallam University, UK.

The dataset used in this project contains over 1 million reviews from the Amazon Apparel category. The analysis includes data preprocessing, text cleaning, sentiment classification, and the application of multiple machine learning models to predict sentiment. The results are visualised using word clouds and confusion matrices, providing insights into customer opinions and the effectiveness of different classification algorithms.

---

## Key Features

- **Data Preprocessing**: Handled missing values, cleaned text data, and merged review headlines with review bodies for comprehensive analysis.
- **Sentiment Classification**: Categorised reviews into positive, neutral, and negative sentiments based on star ratings.
- **Text Cleaning**: Removed stopwords, punctuation, and irrelevant characters to prepare text data for analysis.
- **Feature Extraction**: Applied TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features.
- **Class Imbalance Handling**: Used SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the dataset.
- **Machine Learning Models**: Evaluated the performance of multiple classifiers, including:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (Bernoulli)
  - Support Vector Machine (SVM)
- **Visualisation**: Generated word clouds for positive, neutral, and negative reviews, and plotted confusion matrices for model evaluation.

---

## Dataset

The dataset used in this project is the **Amazon Reviews US Apparel Dataset**, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/krishnalalp/amazon-customer-reviews-dataset-apparel-data). The dataset contains the following columns:
- `marketplace`
- `customer_id`
- `review_id`
- `product_id`
- `product_parent`
- `product_title`
- `product_category`
- `star_rating`
- `helpful_votes`
- `total_votes`
- `vine`
- `verified_purchase`
- `review_headline`
- `review_body`
- `review_date`

**Note**: The dataset should be placed in the same folder as the Jupyter Notebook file for the code to run seamlessly.

---

## Methodology

1. **Data Cleaning**:
   - Handled missing values in the `review_body` column by filling them with the placeholder "Missing."
   - Removed stopwords, punctuation, and special characters from the review text.
   - Merged `review_headline` and `review_body` to create a comprehensive review text for analysis.

2. **Sentiment Classification**:
   - Reviews were classified into three sentiment categories based on star ratings:
     - **Positive**: 4 or 5 stars
     - **Neutral**: 3 stars
     - **Negative**: 1 or 2 stars

3. **Feature Engineering**:
   - Extracted `date`, `month`, and `year` from the `review_date` column for temporal analysis.
   - Applied TF-IDF vectorisation to convert text data into numerical features.

4. **Handling Class Imbalance**:
   - Used SMOTE to balance the dataset by oversampling the minority classes.

5. **Model Training and Evaluation**:
   - Split the dataset into training and testing sets (75% training, 25% testing).
   - Trained and evaluated multiple machine learning models, including Logistic Regression, Decision Tree, KNN, Naive Bayes, and SVM.
   - Generated classification reports and confusion matrices to assess model performance.

6. **Visualisation**:
   - Created word clouds for positive, neutral, and negative reviews to visualise common words in each sentiment category.
   - Plotted confusion matrices to evaluate the performance of each model.

---

## Results

The performance of the machine learning models was evaluated based on accuracy, precision, recall, and F1-score. Below is a summary of the results:

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.721    | 0.76      | 0.72   | 0.72     |
| Decision Tree       | 0.759    | 0.80      | 0.76   | 0.76     |
| KNN                 | 0.732    | 0.79      | 0.73   | 0.74     |
| Naive Bayes         | 0.720    | 0.76      | 0.72   | 0.72     |
| SVM                 | 0.721    | 0.76      | 0.72   | 0.72     |

The **Decision Tree** model achieved the highest accuracy (75.9%), while the **KNN** model had the highest precision (79%) for positive sentiment.

---

## Visualisations

### Word Clouds
- **Positive Reviews**: Common words include "love," "great," "perfect," and "comfortable."
- **Neutral Reviews**: Common words include "okay," "average," and "decent."
- **Negative Reviews**: Common words include "disappointed," "poor quality," and "return."

### Confusion Matrices
- Confusion matrices were plotted for each model to visualise the true vs. predicted sentiment classifications.

---

## Conclusion

This project demonstrates the application of natural language processing (NLP) and machine learning techniques to analyse customer sentiment from Amazon apparel reviews. The results show that machine learning models can effectively classify sentiment, with the Decision Tree model outperforming others in terms of accuracy. The insights gained from this analysis can help businesses understand customer preferences and improve product offerings.

---

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn, WordCloud, Imbalanced-learn
- **Machine Learning Models**: Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM
- **Data Visualisation**: Word Clouds, Confusion Matrices

---

## How to Use This Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/amazon-reviews-sentiment-analysis.git

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/krishnalalp/amazon-customer-reviews-dataset-apparel-data) and place it in the same folder as the Jupyter Notebook file.

4. Run the Jupyter Notebook (amazon_reviews_sentiment_analysis.ipynb) to reproduce the analysis.

---

## Future Work

- Explore deep learning models (e.g., LSTM, BERT) for sentiment analysis.
- Perform topic modeling to identify common themes in reviews.
- Analyse the impact of review length on sentiment classification accuracy.

---

## Contact

For any questions or collaborations, feel free to reach out:

- Name: Krishnalal Purushothaman   
- Email: kaikrishnalal@gmail.com
- LinkedIn: [linkedin.com/in/krishnalalp](https://linkedin.com/in/krishnalalp) 
- GitHub: [github.com/krishnalalp](https://github.com/krishnalalp)

---

Note: This project was completed as part of my Master's degree in Big Data Analytics at Sheffield Hallam University, UK. The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/krishnalalp/amazon-customer-reviews-dataset-apparel-data). Ensure the dataset is placed in the same folder as the Jupyter Notebook file for the code to run seamlessly.

---