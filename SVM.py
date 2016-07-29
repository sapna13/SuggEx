from sklearn.svm import SVC
import numpy as np
from utils import csv
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

trainpath = "data/electronics.csv"
testpath = "data/hotel.csv"

def sentiwordnet_lookup(adjective_list):
    adj_sentiment = []
    for adj in adjective_list:
        adj_synsets = swn.senti_synsets(adj,'a')
        try:
            adj0 = (list(adj_synsets))[0]
        except IndexError:
            adj0 = 'null'
        if(adj0 == 'null'):
            adj_sentiment_entry = [adj, "not found"]
            adj_sentiment.append(adj_sentiment_entry)
        else:
            adj0_senti_scores = [adj0.pos_score(), adj0.neg_score(), adj0.obj_score()]
            adj0_senti_score = max(adj0_senti_scores)
            if(adj0_senti_score==adj0.neg_score()):
                adj0_senti_score = adj0_senti_score*(-1)
            elif(adj0_senti_score==adj0.obj_score()):
                adj0_senti_score = 0

            adj_sentiment_entry = [adj, adj0_senti_score]

            adj_sentiment.append(adj_sentiment_entry)

    return adj_sentiment

#takes a list of texts as input, and returns a list corresponding to the sentiment score of text at the same index
def sentiment_sum(text):
    sentiment_score_sum = []

    for entry in text:
        pos_tagged = nltk.pos_tag(word_tokenize(entry))

        adjective_list = []
        for key,tag in pos_tagged:
            if ((tag == "JJ") or (tag == "JJS") or (tag == "JJR")):
                adjective_list.append(key)

        adjectives_sentiments = sentiwordnet_lookup(adjective_list)

        senti_summation_sentence = 0
        for item in adjectives_sentiments:
            senti_summation_sentence = senti_summation_sentence + item[1]

        sentiment_score_sum.append(senti_summation_sentence)


    return sentiment_score_sum

#data in the form of a list of tuples, where one element is the text, while other is the label
def extract_features(dataset, type):
    # assign lists for the label and features
    text = []
    class_labels = []

    #http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

    vectorizer = CountVectorizer(analyzer='word', binary=False,
                                 decode_error='ignore',
        encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=1000, min_df=1,
        ngram_range=(1, 2), preprocessor=None,
        token_pattern='(?u)\\b\\w\\w+\\b', stop_words=None)


    for i in range(1,len(dataset)-1):
        class_labels.append(dataset[i][1])
        text.append(dataset[i][0])
   #unigram features
    if(type == 'train'):
        feature_vectors = vectorizer.fit_transform(text)

    if(type == 'test'):
        feature_vectors = vectorizer.transform(text)

    else:
        feature_vectors = vectorizer.fit_transform(text)

    feature_array = feature_vectors.toarray()

    selected_features = SelectKBest(chi2, k=300).fit_transform(feature_array, class_labels)


    print(selected_features)
    # sentiment features
   # sentiment_sums = sentiment_sum(text)
   # for j in range(0,len(sentiment_sums)):
           # print(sentiment_sums(j))
           # print (feature_array[j])
            #feature_vectors[j].append(sentiment_sums[j])


    return selected_features, class_labels

# for train-test setting
def trainClassifier(train_vectors, train_labels):

    #feature selection

    svc = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True,
              probability=False, tol=0.001, cache_size=500, class_weight="balanced",
              verbose=True, max_iter=-1, decision_function_shape='ovo', random_state=None)

    svc_model = svc.fit(train_vectors, train_labels)

    return svc_model


if __name__== '__main__':
    trainingData = csv.read_csv(trainpath)
    type="none"
    train_vectors, train_labels = extract_features(trainingData, type)
    testData = csv.read_csv(testpath)

    model = trainClassifier(train_vectors, train_labels)

    #evaluation using cross validation on the training data

    if(type=="train"):
        skf = StratifiedKFold(train_labels, 5)
        train_vectors = np.array(train_vectors)  #change this back for using separate train-test data
        train_labels = np.array(train_labels)   #change this back for using separate train-test data
        for train, test in skf:
            X_train, X_test, y_train, y_test = train_vectors[train], train_vectors[test], train_labels[train], train_labels[test]
            model = trainClassifier(X_train, y_train)
            print(metrics.classification_report(model.predict(X_test) , y_test))


    #evaluation using split of the dataset
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_vectors, train_labels, test_size = 0.4, random_state=42)
    # modelSplit = trainClassifier(X_train, y_train)
    # print(cross_validation.cross_val_score(modelSplit, X_test, y_test, scoring="f1"))


    ##evaluation with the provided test dataset

    if(type=="none"):
        test_vectors, test_labels = extract_features(testData, type)
        modelTrainTest = trainClassifier(train_vectors, train_labels)
        #per class evaluation
        pos_class_vectors = []
        neg_class_vectors = []
        pos_class_label = []
        neg_class_label = []
        for i in range(0,len(test_labels)):
            if(test_labels[1]=="0"):
                neg_class_vectors.append(test_vectors[i])
                neg_class_label.append("0")
            else:
                pos_class_vectors.append(test_vectors[i])
                pos_class_label.append("1")

        print(metrics.classification_report(modelTrainTest.predict(test_vectors), test_labels))

        print(cross_validation.cross_val_score(modelTrainTest, pos_class_vectors, pos_class_label))



