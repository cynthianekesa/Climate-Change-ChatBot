import numpy as np
import random
import json
import pickle
import torch
import re
import gradio as gr
from transformers import BertForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Loading necessary files
with open('./label_mapping.json', 'r') as file:
    label_mapping = json.load(file)

# Loading the BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./save_model')
tokenizer = AutoTokenizer.from_pretrained('./save_model')

# Loading the semantic model and answer embeddings
semantic_model = SentenceTransformer('./sent-transf')
answer_embeddings = np.load('./answer_embeddings.npy')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def semantic_search(query, top_k=1):
    query_embedding = semantic_model.encode([query])
    similarities = cosine_similarity(query_embedding, answer_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Returning the top result as a string
    top_result = list(label_mapping.keys())[top_indices[0]]
    return top_result

def predict_category(question):
    cleaned_question = clean_text(question)
    inputs = tokenizer(f"{cleaned_question}", return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    bert_prediction = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class_id)]
    semantic_results = semantic_search(f"{cleaned_question}")
    if bert_prediction in [result[0] for result in semantic_results]:
        return bert_prediction
    else:
        return semantic_results[0][0]


# Function that Gradio will use to provide chatbot responses
#def chatbot_response(message):
    
    #predicted_category = predict_category(message)
    
    
    #semantic_results = semantic_search(message)
    
    # Check if the predicted category matches any of the top semantic results
    # If it does, use the predicted category; otherwise, use the top semantic result
    #if predicted_category in [result[0] for result in semantic_results]:
        #selected_category = predicted_category
    #else:
        #selected_category = semantic_results[0][0]
    
    # Retrieve the response based on the selected category
    # Assuming label_mapping contains the responses mapped to categories
    #response = label_mapping.get(selected_category, "Sorry, I don't understand.")
    
    #return response
def chatbot_response(message):
    semantic_results = predict_category(message)
    response = semantic_search(message)
    return response


# Creating the Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs="text",
    outputs="text",
    title="EcoSage(The ClimateChangeChatBot)",
    description="A BERT model chatbot that provides insightful answers, practical solutions, and up-to-date information to empower individuals in the fight against climate change.",
    theme="huggingface",
)

# Launching the Gradio interface
iface.launch()