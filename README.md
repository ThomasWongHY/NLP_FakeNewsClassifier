# NLP_FakeNewsClassifier

This is a challenge from StackUp about NLP techniques in Python

## 1: NLP Basics for Text Preprocessing
### Tokenization (to divide strings into sentences or words).
```python
sample = 'Once upon a time, there was a little girl who loved to dance. She would spin and twirl around her room every day, dreaming of becoming a ballerina. One day, a famous ballet teacher saw her dancing and offered to train her. From then on, the little girl\'s dreams came true as she danced on stages all around the world.'
sentence_tokens = sent_tokenize(sample)
word_tokens = word_tokenize(sample)
```

### Removal of stop words
```python
stopwords = set(stopwords.words('english'))
stopwords_removed = [i for i in word_tokens if i not in stopwords]
```

### Stemming and Lemmatization (stemming removes the suffix from the word while lemmatization takes into account the context and what the word means in the sentence.)
```python
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()

sample_stem = [stemmer.stem(token) for token in stopwords_removed]
sample_lemma = [lemma.lemmatize(token) for token in stopwords_removed]
```

## 2: Vectorization
### Bag-of-Words Model (BOW)
```python
count_vectorizer = CountVectorizer()
bow = count_vectorizer.fit_transform(sample_text)
```

### Term Frequency-Inverse Document Frequency (TF-IDF)
```python
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(text_data)
```

### Word Embeddings
```python
tokens_in_text = [nltk.word_tokenize(sentence.lower()) for sentence in text_data] 
model = Word2Vec(tokens_in_text, min_count=1)
vector = model.wv['class']
```

## 3: Fake News Classifier

### Data Exploration
```python
df.head()
df.shape
df.info()
```

### Data Preprocessing
```python
# Check for null values and if any, fill them with an empty string 
df.isnull().sum()
df.isnull().sum()
df.fillna(value="", axis=1)

# Ensure the data type of column 'text' is str
df['text'] = df['text'].astype(str)

# Define a function to tokenize the text given
def tokenize_text(text):
    return word_tokenize(text)

# Apply the tokenize_text function to the 'text' column of the DataFrame and create a new column 'tokenized_text'
df['tokenized_text'] = df["text"].apply(tokenize_text)

# Define stop words
stop_words = set(stopwords.words('english'))

# Define a function to remove stopwords from a list of tokens
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    tokens_without_stopwords = [word for word in tokens if word.lower() not in stop_words]     # replace and fill in underscores
    return tokens_without_stopwords

# Apply the remove_stopwords function to the 'tokenized_text' column
df['stopwords_removed'] = df['tokenized_text'].apply(remove_stopwords)
```

### Data Preparation
```python
# Separate the data into features and targets
X_df = df['tokenized_text']
y_df = df['label']

# convert the label column into a numerical one.
y_df = y_df.astype(int)

# Perform vectorization using the TFIDF Vectorizer and fit and transform the tokenized documents
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(X_df)
```

### Model Building
```python
# Split the data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, y_df, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
```

### Model Evalution
```python
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
```
