# Mini-projet

## By: HAMMOUDA Med Amine

## The problem and the objectives

The problem addressed in the paper is the classification of news articles into different groups. The objective of the paper is to propose a new approach based on TF-IDF and SVM that achieves high classification precision. Specifically, the paper aims to classify news into various groups so that users can identify the most popular news group in the desired country at any given time. The proposed approach is evaluated using two datasets, the BBC dataset and the 20Newsgroup dataset, and achieves high classification precisions of 97.84% and 94.93% respectively.

## State of the art on which the work was based

The state of the art on which the work was based is not explicitly stated in the given texts. However, the paper mentions that the proposed approach was evaluated and compared with other classification methods, indicating that the authors were likely aware of existing methods in the field. Additionally, the paper cites several references, including studies on news classification using different techniques such as the hidden Markov model and decision trees. These references suggest that the authors were familiar with the existing literature on news classification.

## Research methodology followed by the paper

The research methodology followed by the paper involves proposing a new approach for news classification based on TF-IDF and SVM, and evaluating its performance using two datasets, the BBC dataset and the 20Newsgroup dataset. The proposed approach consists of three steps: text preprocessing, feature selection based on TF-IDF, and text classification using SVM. The paper describes each step in detail and provides a flowchart of the proposed method. The performance of the proposed approach is evaluated using precision and F-score metrics, and compared with other classification methods. The paper also discusses the results and limitations of the proposed approach, and provides suggestions for future research. Overall, the research methodology followed by the paper is a combination of proposing a new approach, implementing it, and evaluating its performance using established metrics and datasets.

## Techniques used

The techniques used in the paper are the TF-IDF (Term Frequency-Inverse Document Frequency) method and Support Vector Machine (SVM) for news article classification. The TF-IDF method is used to extract features from news articles, while SVM is used for classifying articles into different groups. The paper also describes the necessary text preprocessing steps for cleaning and normalizing the data before feature extraction and classification. Additionally, the paper uses precision and F-score metrics to evaluate the performance of the proposed approach and compares it with other classification methods.

## Process

### Import the necessary libraries:


```python
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report, accuracy_score
```

### Download NLTK resources:


```python
nltk.download('stopwords')
nltk.download('punkt')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\hammouda\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\hammouda\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    




    True



### Define functions for data preprocessing:

- We have defined three functions for data preprocessing. The preprocess_data function takes a piece of data and performs several cleaning operations. We remove specific patterns like email headers, subjects, and other email-related information. We also remove email addresses, special characters, and punctuation. Furthermore, we convert the text to lowercase. Additionally, there may be some further steps indicated by "..." in the code snippet that are not explicitly mentioned. Finally, we return the preprocessed data.

- The remove_stopwords function removes stopwords from the input text. We initialize a set of English stopwords using NLTK's stopwords.words('english'). Then, we tokenize the input text into individual words using NLTK's word_tokenize function. We filter out the stopwords from the tokens and join the remaining words back into a single string. The resulting string contains the text without stopwords.

- The tokenize function tokenizes the input text into individual words using NLTK's word_tokenize function. It returns a list of tokens representing the words in the text.


```python
def preprocess_data(data):
    data = re.sub(r'From:.*\n', '', data)
    data = re.sub(r'Subject:.*\n', '', data)
    # ... (remove email headers and footers)
    data = re.sub(r'\S+@\S+', '', data)  # Remove email addresses
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)  # Remove special characters and punctuation
    data = data.lower()  # Convert to lowercase
    # ... (remove extra whitespace, remove digits)
    return data

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word.casefold() not in stop_words]
    return ' '.join(filtered_text)

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens
```

### Load and preprocess the dataset:

- We have specified a list of categories consisting of 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', and 'talk.religion.misc'. Then, using the fetch_20newsgroups function from the sklearn.datasets module, we have obtained the newsgroups dataset with the subset set to 'all' and filtered based on the specified categories. The data has been shuffled and a random state of 42 has been set for reproducibility.

- Next, we have split the dataset into training and testing sets using the train_test_split function from the sklearn.model_selection module. The training data, testing data, training targets, and testing targets have been assigned to the variables train_data, test_data, train_target, and test_target, respectively.

- To preprocess the training data, we have applied the preprocess_data function to each document in the train_data list using a list comprehension. Then, we have removed stopwords from the preprocessed data by applying the remove_stopwords function to each document in preprocessed_train_data. Finally, we have tokenized the preprocessed data by applying the tokenize function to each document in preprocessed_train_data.

- Similarly, we have preprocessed the test data by applying the preprocess_data, remove_stopwords, and tokenize functions to each document in the test_data list, resulting in the preprocessed_test_data list.


```python
categories = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
newsgroups_data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
train_data, test_data, train_target, test_target = train_test_split(newsgroups_data.data, newsgroups_data.target, test_size=0.2, random_state=42)

preprocessed_train_data = [preprocess_data(d) for d in train_data]
preprocessed_train_data = [remove_stopwords(d) for d in preprocessed_train_data]
preprocessed_train_data = [tokenize(d) for d in preprocessed_train_data]

preprocessed_test_data = [preprocess_data(d) for d in test_data]
preprocessed_test_data = [remove_stopwords(d) for d in preprocessed_test_data]
preprocessed_test_data = [tokenize(d) for d in preprocessed_test_data]
```

### Convert the preprocessed data into TF-IDF features:

- To further process the preprocessed training data, we have joined the tokenized words in each document of the preprocessed_train_data list into a single string. This is done using a list comprehension where we apply the join function with a space as the separator to concatenate the tokens in each document. The resulting list is assigned to the variable train_text.

- Similarly, we have performed the same step for the preprocessed test data by joining the tokenized words in each document of the preprocessed_test_data list into a single string. The resulting list is assigned to the variable test_text.

- Next, we have created an instance of the TfidfVectorizer class from the sklearn.feature_extraction.text module. This vectorizer will be used to convert the text data into TF-IDF features.

- For the training data, we have fit the vectorizer on the train_text data by calling the fit_transform method of the vectorizer object, which fits the vectorizer to the training data and transforms the training data into TF-IDF features. The resulting matrix of features is assigned to the variable X_train.

- For the test data, we have transformed the test_text data into TF-IDF features using the previously fitted vectorizer. This is done by calling the transform method of the vectorizer object on the test_text data. The resulting matrix of features is assigned to the variable X_test.


```python
train_text = [' '.join(doc) for doc in preprocessed_train_data]
test_text = [' '.join(doc) for doc in preprocessed_test_data]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)
```

### Train and evaluate the classifier:

- To perform classification using the Support Vector Machine (SVM) algorithm, we have instantiated an instance of the NuSVC class from the sklearn.svm module, specifying the 'rbf' (radial basis function) kernel. This kernel is commonly used for SVM classification tasks. The SVM classifier is assigned to the variable svm_classifier.

- We then trained the SVM classifier on the training data by calling the fit method of the svm_classifier object, passing the training features (X_train) and training targets (train_target) as arguments. This step allows the SVM classifier to learn from the training data and build a model.

- After training the classifier, we made predictions on the test data by calling the predict method of the svm_classifier object, passing the test features (X_test) as an argument. The predictions are assigned to the variable y_pred.

- To evaluate the performance of the classifier, we printed a classification report by calling the classification_report function from the sklearn.metrics module, passing the true test targets (test_target) and the predicted targets (y_pred) as arguments. This report provides metrics such as precision, recall, F1-score, and support for each class.

- Additionally, we printed the accuracy score by calling the accuracy_score function from the sklearn.metrics module, passing the true test targets (test_target) and the predicted targets (y_pred) as arguments. The accuracy score represents the proportion of correctly classified instances in the test data.

- Overall, these steps allow us to train an SVM classifier on the TF-IDF features extracted from the preprocessed training data and evaluate its performance on the preprocessed test data.


```python
svm_classifier = NuSVC(kernel='rbf')
svm_classifier.fit(X_train, train_target)
y_pred = svm_classifier.predict(X_test)

print("Classification Report:")
print(classification_report(test_target, y_pred))
print("Accuracy:", accuracy_score(test_target, y_pred))
```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.88      0.97      0.93       186
               1       0.98      0.98      0.98       190
               2       0.94      0.88      0.91       156
               3       1.00      0.92      0.96       119
    
        accuracy                           0.94       651
       macro avg       0.95      0.94      0.94       651
    weighted avg       0.95      0.94      0.94       651
    
    Accuracy: 0.9431643625192012
    

## Conclusion 

In this code, we performed text classification using the Support Vector Machine (SVM) algorithm on a subset of the 20 Newsgroups dataset. We preprocessed the text data by removing email headers, footers, addresses, special characters, punctuation, and stopwords. Then, we transformed the preprocessed text into numerical features using TF-IDF representation. We trained an SVM classifier on the transformed training data and made predictions on the test data. Finally, we evaluated the performance of the classifier using a classification report and accuracy score. Overall, this code demonstrated the process of preprocessing text data, transforming it into numerical features, training an SVM classifier, and evaluating its performance for text classification.
