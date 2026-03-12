from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# training data
emails = [
    "Win money now",
    "Claim your free prize",
    "Limited offer click now",
    "Hi how are you",
    "Let's meet tomorrow",
    "Project meeting today"
]

labels = [1,1,1,0,0,0]   # 1 = spam , 0 = not spam

# convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# train model
model = MultinomialNB()
model.fit(X, labels)

# user input
msg = input("Enter email text: ")

# prediction
msg_vector = vectorizer.transform([msg])
result = model.predict(msg_vector)

if result[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")