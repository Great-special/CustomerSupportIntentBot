import streamlit as st
import joblib
import pandas as pd

df = pd.read_csv('customer-intent-dataset/intent-responses.csv')
responses = df.set_index('Intent').to_dict().get('Response')

# Load model and vectorizer
model = joblib.load("packages/model.pkl")
vectorizer = joblib.load("packages/vectorizer.pkl")


# Streamlit UI
st.title("Customer Support Chatbot")
st.write("Enter your query below:")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Say something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Vectorize input and predict
    user_input_vec = vectorizer.transform([prompt])
    prediction = model.predict(user_input_vec)[0]
    response = responses.get(prediction, "Sorry, I did not understand that. Please try again.")

    st.session_state.messages.append({"role": "bot", "content": response})


# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("bot").write(message["content"])

