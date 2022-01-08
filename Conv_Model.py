import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from parsivar import Normalizer, SpellCheck
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# !Space Correction & Pinglish Convertor
def normalize(text):
    normalizer = Normalizer(pinglish_conversion_needed=True)
    text = normalizer.normalize(text)
    return text


# !Spell Check
def spell_check(text):
    spell_checker = SpellCheck()
    text = spell_checker.spell_corrector(text)
    return text


# !Remove StopWord
def stop_word(text):
    tokens = text.split()
    stop_words = set(stopwords.words('persian'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = ' '.join(tokens)
    return tokens


# !Reading a File
log = pd.read_csv(r'F:\sourcecode\pardazesh01\data.csv', index_col=False)
# log = log[:10]

# !Applying Clean Function
max_len = 100
log['clean'] = log['Text'].apply(lambda x: normalize(x))
# log['clean'] = log['clean'].apply(lambda x: spell_check(x))
# log['clean'] = log['clean'].apply(lambda x: stop_word(x))
log['clean'] = log['clean'].apply(lambda x: x[:max_len])

# !Creating Uniform Data
log_list = [list(x) for x in zip(log['clean'], log['Suggestion'])]
pos = list(filter(lambda x: x[1] == 1, log_list))
un = list(filter(lambda x: x[1] == 2, log_list))
neg = list(filter(lambda x: x[1] == 3, log_list))

log_list = pos[:450]+neg[:450]+un[:450]
# log_list = pos + neg + un

# !Train and Test Split
X, Y = zip(*log_list)
trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.9)

# !Tokenizing Text
tk = Tokenizer()
tk.fit_on_texts(trainX)
vocab_size = len(tk.word_index) + 1
encode_trainX = tk.texts_to_sequences(trainX)
encode_testX = tk.texts_to_sequences(testX)
encode_trainX = pad_sequences(encode_trainX, maxlen=max_len)
encode_testX = pad_sequences(encode_testX, maxlen=max_len)

# !Vectoring Tag
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)
encode_trainY = to_categorical(np.array(trainY))
encode_testY = to_categorical(np.array(testY))
num_cat = encode_trainY.shape[1]
# print(encode_trainY)
# print(encode_testY)
# print(log_list)
# log.to_excel(r'data1.xlsx', index=False)

# !Creating Model (82%)
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_len))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(num_cat, activation='softmax'))
print(model.summary())

# Compiling Network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting Network
m = model.fit(encode_trainX, encode_trainY, epochs=20, verbose=2)

# Evaluating Network
loss, acc = model.evaluate(encode_testX, encode_testY, verbose=0)
print('Test Accuracy: %f' % (acc * 100))
print('Test loss: %f' % loss)
