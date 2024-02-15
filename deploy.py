import numpy as np
import pandas as pd
import streamlit as st
from scripts.data_preparation import clean_text, preprocess_text, extract_features, make_prediction

def main():
    st.header("Customer Sentiment Analysis")
    st.markdown("""
    :dart: AI Application to classify customer sentiments.
""")
    
    st.sidebar.image('./slide-img.jpg')
    st.sidebar.info('This application is for analysing sentiments.')
    selection = st.sidebar.selectbox("Select how you want to do predicion?", ["Online", "Batch"])


    if selection=="Online":
        st.write("Write the text to analyse sentiment: ")
        text = st.text_area("Customer Feedback or any text to analyse the sentiment...")
        if st.button("Predict"):
            text = clean_text(text)
            text = preprocess_text(text)
            features = extract_features([text])
            prediction = make_prediction(features)
            prediction = prediction[0]
            if prediction == 0:
                st.info("Text has neutral sentiment.")
            elif prediction == -1:
                st.warning("Text has negative sentiment.")
            else:
                st.success("Text has positive sentiment.")
    else:
        uploaded_file = st.file_uploader("Select a file...")
        if uploaded_file:
            texts = pd.read_excel(uploaded_file)
            df = texts.copy()
            if st.button("Predict"):
                df['Text'] = df['Text'].apply(clean_text)
                df['Text'] = df['Text'].apply(preprocess_text)
                features = extract_features(df['Text'])
                predictions = make_prediction(features)
                predictions = pd.DataFrame({'Text': texts['Text'], 'Sentiment': predictions})
                predictions['Sentiment'] = predictions['Sentiment'].replace({-1: "negative", 0: "neutral", 1: "positive"})
                st.write(predictions)

if __name__=="__main__":
    main()