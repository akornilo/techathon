import pickle

# Load the relevant files
model = pickle.load(open("movie_review_model.pkl"))
vectorizer = pickle.load(open("movie_review_vectorizer.pkl"))

print "Models loaded"
while True:
    print "Enter a movie review:  ",

    text = raw_input()

    text = text.strip()

    text_vector = vectorizer.transform([text])

    num_pred = model.predict(text_vector)[0]

    if num_pred == 1:
        print "I think it's good"
    else:
        print "I think it's bad"
    print