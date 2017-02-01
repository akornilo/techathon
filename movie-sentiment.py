#!/usr/bin/env python
# coding: utf-8
import argparse
import cPickle
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def train_clf(vectorizer_cls, feature_class_fn, train_filename):
    f = open(train_filename)

    cur = -1
    all_X = []
    all_y = []
    for line in f:
        if cur == -1: # Skip first line
            cur += 1
            continue
        parts = line.split('\t')
        if cur < int(parts[1]):
            all_X.append(parts[2])
            all_y.append(int(parts[3]))
            cur += 1

    all_X2 = []
    all_y2 = []
    all_bad = []
    all_good = []

    for i in range(len(all_X)):
        if all_y[i] == 2:
            continue

        new_y = -1 if feature_class_fn(all_y[i]) else 1

        all_y2.append(new_y)
        all_X2.append(all_X[i])

        if new_y == -1:
            all_bad.append(all_X[i])
        else:
            all_good.append(all_X[i])

    vectorizer = vectorizer_cls(min_df=5, stop_words='english', ngram_range=(1, 3))
    vX = vectorizer.fit_transform(all_X2)

    X_train, X_test, y_train, y_test = train_test_split(vX, all_y2, test_size=0.2)

    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)

    return clf, vectorizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', choices=('count', 'tfid'), default='count')
    parser.add_argument('-p', default='clf.pickle')
    parser.add_argument('-f', default='train.tsv')
    parser.add_argument('review', nargs='*')
    args = parser.parse_args()

    try:
        with open(args.p) as fp:
            print 'Loading classifier...'
            clf, vectorizer = cPickle.load(fp)
    except IOError:
        print 'Training classifier...'
        vectorizers = {'count': CountVectorizer, 'tfid': TfidfVectorizer}
        clf, vectorizer = train_clf(vectorizers[args.V], lambda v: v < 2, args.f)
        print 'Saving classifier...'
        with open(args.p, 'w') as fp:
            cPickle.dump((clf, vectorizer), fp)

    if args.review:
        print 'Predicting...'
        classification = clf.predict(vectorizer.transform([' '.join(args.review)]))
        print 'POSITIVE' if classification[0] == 1 else 'NEGATIVE'


if __name__ == '__main__':
    main()
