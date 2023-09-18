import numpy as np
import pickle
import streamlit as st

#loading the saved model
model = pickle.load(open('models/trained_model.sav','rb'))
loaded_vectorizer = pickle.load(open('models/vectorizer.pickle', 'rb'))

# creating a function for classification
def spam_detection(input_data):

    a = model.predict(loaded_vectorizer.transform([input_data]))
    if a[0]== 1:
        return "It's a spam mail."
    else:
        return "It's not a spam mail."
    
def main():
    #title
    st.title('Spam-mail-detection-WebApp')
    #input data
    email_text = st.text_area('Enter an email text:', '')
    #code for prediction
    res = ''

    # creating a button
    if st.button('Classify'):
        res = spam_detection(email_text)

    st.success(res)

if __name__ == '__main__':
    main()

