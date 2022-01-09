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
import emojies
import matplotlib.pyplot as plt


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


# !Convert Emojis
def convert_emojis(text):
    text = emojies.replace(text)
    return text


# !Reading a File
# log = pd.read_csv(r'F:\sourcecode\pardazesh01\data.csv', index_col=False)
log = pd.read_excel(r'F:\sourcecode\pardazesh01\1e4_S_01.xlsx', index_col=False)
# log = log[:2000]

# create figure and axis
fig, ax = plt.subplots()
ax.hist(log['tag'])
plt.show()
# !Applying Clean Function
max_len = 200
log['clean'] = log['Text'].apply(lambda x: convert_emojis(x))
log['clean'] = log['clean'].apply(lambda x: normalize(x))
# log['clean'] = log['clean'].apply(lambda x: spell_check(x))
log['clean'] = log['clean'].apply(lambda x: stop_word(x))
log['clean'] = log['clean'].apply(lambda x: x[:max_len])

# !Creating Uniform Data
log_list = [list(x) for x in zip(log['clean'], log['tag'])]

# pos = list(filter(lambda x: x[1] == 1, log_list))
# un = list(filter(lambda x: x[1] == 2, log_list))
# neg = list(filter(lambda x: x[1] == 3, log_list))

# !Filter Category
khs = list(filter(lambda x: x[1] == 5, log_list))
khn = list(filter(lambda x: x[1] == 6, log_list))
shd = list(filter(lambda x: x[1] == 8, log_list))
ghm = list(filter(lambda x: x[1] == 9, log_list))

# log_list = pos[:450]+neg[:450]+un[:450]
# log_list = pos + neg + un
# log_list = khs[:900]+khn[:900]+shd[:900]+ghm[:900]
log_list = khs[:2000]+shd[:2000]

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
m = model.fit(encode_trainX, encode_trainY, epochs=50, verbose=2)

# Evaluating Network
loss, acc = model.evaluate(encode_testX, encode_testY, verbose=0)
print('Test Accuracy: %f' % (acc * 100))
print('Test loss: %f' % loss)
# !khashm, khonsa, shadi, gham of 1e4
# model.save('s_1e4.model')
# !khashm, shadi of 1e4 83%
model.save('s_2_1e4.model')

