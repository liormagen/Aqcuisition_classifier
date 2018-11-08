import pandas as pd
from operator import itemgetter
from collections import Counter
from urllib.parse import urlparse
from keras.preprocessing.text import text_to_word_sequence
from nltk import SnowballStemmer, re, downloader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

file_name = ...


# What the Data tells? Most of the class 0 URLs that Iv'e checked are leading to nothing (404) I think that taking
# only the description as input feature is enough, the knowledge if there was an acquisition or not supposed to be
# there and other feature may confuse the model (I proved that this assumption isn't right, there are some cases
# where the title is not enough and it talks about the acquisition only in the content!!!!)
# The content always (or in most cases) starts with the title text exactly.

# Summary of my work - Iv'e tried to use KFolds where k=3 and k=5 but it didn't improve the results (I'm focusing on
# F1), in addition Iv'e tried to find correlation between titles that contain the words "acquire",
# "acquisition", "join"... but this kind of correlation doesn't exist (those words appear equally in positive and
# negative classes documents.
#
# I thought about language detection (I saw some Chinese documents) but the accuracy is
# seems high enough and the amount of dedicated features will probably ignore such words because of their low
# frequency in the data.
#
# Most of the negative class' URLs led to an empty web site (message 404). This is a knowledge
#  I could exploit but, again, the results seemed good enough in this case.
#
# I used only the title and the URL for each document, I found that the content and the description damage my
# results. There are some private cases where the title is not enough as an input but it did not justify using it
# though.
#
# Iv'e tried to use various architectures but in the end the Linear SVM produced the best results (from my
# experience it works really good with short texts).

class Preprocessor(object):
    def __init__(self, lang='english', test_size=.2):
        self.lang = lang
        self.test_size = test_size
        self.sw = self.clean_sw()

    def english_stemmer(self, word):
        stemmed_word = SnowballStemmer(self.lang).stem(word)
        return stemmed_word

    def clean_sw(self):
        try:
            sw = stopwords.words(self.lang)
        except LookupError:
            downloader.download('stopwords')
            sw = stopwords.words(self.lang)
        return set([self.english_stemmer(w) for w in sw])

    @staticmethod
    def strip_url(text, return_count=False):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if return_count:
            return len(urls)
        for url in urls:
            text = text.replace(url, '_URL_')
        text = text.replace('https:', '')
        return text

    def preprocessor_flow(self, sentence):
        tokens = [self.english_stemmer(w) for w in text_to_word_sequence(sentence, lower=True)]
        return [w for w in tokens if w not in self.sw]

    # def cross_val(self, x_train_, y_train_):
    def cross_val(self, x_train_):
        return train_test_split(x_train_, test_size=self.test_size, random_state=47)


class Classifier(object):
    def __init__(self, max_len=200, max_features=500, lowercase=False, cls_model='svm'):
        self.max_len = max_len
        self.max_features = max_features
        self.lowercase = lowercase
        self.input_vectorizer = CountVectorizer(tokenizer=self.trivial_tokenizer, max_features=self.max_features,
                                                lowercase=self.lowercase)
        self.tfidf = TfidfTransformer()
        cls_models = {'svm': LinearSVC(), 'xgb': XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)}
        self.model = cls_models[cls_model]

    @staticmethod
    def trivial_tokenizer(text):
        return text

    def train(self, x_train, y_train):
        print('Training classifier...')
        pipeline = [('Vectorizer', self.input_vectorizer), ('TFIDF', self.tfidf), ('cls_model', self.model)]
        self.model = Pipeline(pipeline)
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        print('Predicting...')
        y_predicted = self.model.decision_function(x_test)
        return y_predicted

    @staticmethod
    def cls_report(y_true, y_predict):
        print(classification_report(y_true, y_predict, digits=3))


if __name__ == '__main__':
    # ---Read data---
    df = pd.read_csv(file_name)
    rows = df.values.tolist()

    neg_class = .0
    pos_class = 1.

    # The ratio between the amount of positive and negative classes
    class_dist = df['class'].value_counts().to_dict()
    if class_dist[neg_class] == 0:
        class_dist[neg_class] = 1
    pos_neg_ratio = (class_dist[pos_class]/class_dist[neg_class])
    print('%s positive class documents\n%s negative class documents\nThe ratio between positive and negative '
          'documents is: %0.3f' % (class_dist[pos_class], class_dist[neg_class], pos_neg_ratio))

    # The ratio between the amount of positive and negative classes is ~0.12. In such case there will be a significant
    # bias towards the negative class, we should overcome this by balancing our data
    max_negative_counter = 7000

    # ---Preprocess data---
    prep = Preprocessor()
    x_train_ = []
    y_train_ = []
    ignore_idxs = []
    urls = []
    acquisition_docs = []
    original_sentences = []
    print("%s documents pre filtration" % df.values.__len__())

    # We'll find null values' indexes and ignore them
    negative_counter = 0
    nan_idxs = df[df.isnull().T.any().T].index.tolist()
    for idx, input_row in enumerate(rows):
        if not isinstance(input_row[2], str) or not isinstance(input_row[5], float) or idx in nan_idxs:
            ignore_idxs.append(idx)
            continue
        if input_row[5] == 0.:
            if negative_counter >= max_negative_counter:
                continue
            negative_counter += 1
        # We'll concat the articles' titles & URL and use it as an input
        url = urlparse(input_row[1]).netloc
        x_train_.append(prep.preprocessor_flow(input_row[2] + ' ' + input_row[1]))
        y_train_.append(input_row[5])
        original_sentences.append(input_row[2] + ' ' + input_row[1])
        urls.append(url)

    if x_train_.__len__() != y_train_.__len__():
        raise Exception('Mismatch between inputs lengths')

    print("%s documents post filtration" % x_train_.__len__())
    print('Train set (post balancing) contains %s acquisition related articles and %s non acquisition related articles' %
          (y_train_.count(pos_class), y_train_.count(neg_class)))

    neg_pos_urls = {str(pos_class): [], str(neg_class): []}
    for idx, url in enumerate(urls):
        neg_pos_urls[str(y_train_[idx])].append(url)

    urls_hist = {str(pos_class): Counter(neg_pos_urls[str(pos_class)]), str(neg_class):
                 Counter(neg_pos_urls[str(neg_class)])}

    # A list of unique URLs that appeared only in the negative class URLs
    only_neg_urls = list(set(neg_pos_urls[str(neg_class)]).difference(neg_pos_urls[str(pos_class)]))

    # ---Train/Test split---
    x_idxs = list(range(len(x_train_)))
    # x_train, x_test, y_train, y_test = prep.cross_val(x_train_, y_train_)
    x_train_idxs, x_test_idxs = prep.cross_val(x_idxs)
    x_train, y_train = itemgetter(*x_train_idxs)(x_train_), itemgetter(*x_train_idxs)(y_train_)
    x_test, y_test = itemgetter(*x_test_idxs)(x_train_), itemgetter(*x_test_idxs)(y_train_)
    # x_train_idxs, x_test_idxs, y_train_idxs, y_test_idxs = prep.cross_val(x_idxs, x_idxs)

    # ---Classifying---
    cls = Classifier()
    cls.train(x_train, y_train)
    # y_predict = cls.predict(x_test)
    y_predict_score = cls.predict(x_test)
    proba_average = y_predict_score.sum() / len(y_predict_score)
    y_predict = [1.0 if x >= .0 else 0.0 for x in y_predict_score]
    cls.cls_report(y_test, y_predict)

    for idx, (y_true, y_pred) in enumerate(zip(y_test, y_predict)):
        if y_true != y_pred:
            print('Predicted: %s, True: %s, Score: %0.3f\n%s\n' % (y_pred, y_true, y_predict_score[idx],
                                                                original_sentences[x_test_idxs[idx]]))
    # ---Analyze mistakes
    # Here I noticed that the train set isn't perfect
    # The sentence "Ellipse Technologies Acquired by NuVasive for $380 Million Cash - NASDAQ.com" talks about an
    # aqcuisition while its y_true=0. In most cases train set are not perfect.
    # More examples:

    # Predicted: 1.0, True: 0.0, Score: 1.4279117873899703 Employee-Owned System Scale Acquires Denham Springs’ 
    # Central Scales & Control http://www.prnewswire.com/news-releases/employee-owned-system-scale-acquires-denham
    # -springs-central-scales--control-300261626.html 

    # Predicted: 1.0, True: 0.0, Score: 1.392248169061919 Ervin Cohen And Jessup Acquires Land Use Group 
    # http://www.prnewswire.com/news-releases/ervin-cohen-and-jessup-acquires-land-use-group-300269599.html 
