# Climate-Change-ChatBot
## 1. **Introduction**
This project implements a domain-specific Climate Change Chatbot using a pre-trained language model fine-tuned on a climate change-related dataset. The chatbot is designed to answer questions and provide accurate contextual information about climate change, its impacts, and potential solutions for improved climate change awareness and empowerment of individuals in its fight. It leverages Hugging Face's `transformers` library and TensorFlow for fine-tuning and deployment.

---

## 2. **Dataset Collection and Preprocessing**
The dataset used for training the chatbot consists of **conversational pairs** related to climate change. Features inlcude 'answer' and 'question' with 1277 rows. questions and answers on topics such as global warming, renewable energy, carbon emissions, and climate policies. Here is the link to the dataste [https://huggingface.co/datasets/Vinom/ClimateChangeQA]

**Preprocessing Steps:**
- **Tokenization**: The text data was tokenized using 'bert-base-uncased' pre-trained model suitable for conversational chatbots.
- **Normalization**: Text was normalized by converting to lowercase, removing special characters, and handling contractions.
- **Dataset Splitting**: The dataset was split into training, validation, and test sets
- **Encoding**:The tokenized text was encoded into input IDs and attention masks, which are required by the model for training and inference.


---

## 3. **Model Selection and Fine-Tuning**
A pre-trained Bert transformer was used due to its strong performance in natural language understanding and generation tasks.

**Fine-Tuning Process:**
- **Model Initialization**: The pre-trained model was loaded using Hugging Face's `transformers` library.
- **Hyperparameter Tuning**: Key hyperparameters were tuned, including:
  - **Learning Rate**: Tested values between 1e-5 and 5e-5.
  - **Batch Size**: Experimented with batch sizes of 16, 32, and 64.
  - **Optimizer**: AdamW optimizer was used with weight decay.
  - **Training Epochs**: Trained for 3–5 epochs to avoid overfitting.
- **Training**: Training progress was monitored using validation loss and accuracy.

**Hyperparameter Insights:**
- A lower learning rate (e.g., 2e-5) resulted in more stable training and better convergence.
- A batch size of 32 provided a good balance between training speed and memory usage.
- A higher weight decay (e.g 0.1) helps regularize the model and prevent overfitting, while a lower decay(e.g 0.001) is useful if the model is underfitting.
- Training for 4 epochs yielded the best performance without overfitting.
  

---

## 4. **Evaluation**

**Quantitative Metrics:**
- **BLEU Score**: Used to evaluate the quality of generated responses by comparing them to reference answers.
- **F1-Score**: Measured the chatbot's ability to correctly classify and respond to user intents.
- **Perplexity**: Assessed the model's confidence in generating responses.
  
- ![image](https://github.com/user-attachments/assets/d3ad7224-8ef2-4177-a387-d7ceded3db74)
  
**Metric	Score	Interpretation**
- *BLEU_ 0.433_ Generated responses have a 43.3% overlap with the reference answers in terms of n-grams (e.g., 1-gram, 2-gram). This shows a decent overlap with reference, but responses could be more precise.*
- *ROUGE_ -1	0.643_ 64.3% of the words in the reference text are also present in the model’s output.	Strong overlap of key terms, indicating good relevance.*
- *ROUGE_ -L	0.571_	 The model is capturing 57.1% of the key phrases or sequences from the reference. This shows good alignment of phrases, but fluency and coherence could be improved.*
- *Perplexity_	10.076_	Low perplexity, indicating the model is confident in its predictions.*


**Qualitative Testing:**
![chatbot 1](https://github.com/user-attachments/assets/479c8cb0-5be3-4cb3-a947-3658f6309181)

![chatbot 2](https://github.com/user-attachments/assets/66086f18-41fb-4171-a1cd-f6098464dd89)

![chatbot 3](https://github.com/user-attachments/assets/7ce26e3b-46cd-40fc-ad8f-6267f733e5d0)

![chatbot 4-semantic search relevance](https://github.com/user-attachments/assets/3a49b981-8ef5-4b9e-a677-2632e658834d)


---

## 5. **Deployment**
The chatbot was deployed using **Gradio**, a simple and intuitive web interface framework. 

### Installation
To install the required packages, run:

```bash
pip install -r requirements.txt






---

#### 6. **GitHub Repository**
The project is hosted on GitHub, with the following structure:
- **Notebooks/**: Contains the Jupyter Notebook for data preprocessing, model training, and evaluation.
- **Scripts/**: Includes Python scripts for fine-tuning and deployment.
- **Models/**: Stores the fine-tuned model weights.
- **README.md**: Provides an overview of the project, dataset details, performance metrics, and instructions for running the chatbot.
- **Examples/**: Contains sample conversations with the chatbot.

**README.md Contents:**
- **Dataset**: Description of the dataset used for training.
- **Performance Metrics**: BLEU score, F1-score, and perplexity results.
- **Steps to Run the Chatbot**: Instructions for setting up the environment and running the chatbot.
- **Example Conversations**: Demonstrations of the chatbot's responses to various queries.

---

#### 7. **Demo Video**
A **3–5 minute demo video** was created to showcase the chatbot's functionality. The video includes:
- **Introduction**: Brief overview of the project and its goals.
- **User Interaction**: Demonstration of the chatbot answering climate change-related questions.
- **Key Insights**: Discussion of the chatbot's performance, challenges faced, and future improvements.

---

#### 8. **Conclusion**
The Climate Change Chatbot successfully leverages a pre-trained Transformer model to provide accurate and relevant responses to user queries. Fine-tuning the model on a domain-specific dataset and deploying it using Gradio and Flask ensures that the chatbot is both functional and accessible. The project demonstrates the potential of Transformer models in building domain-specific conversational agents and highlights the importance of careful dataset preparation and hyperparameter tuning.

**Future Work:**
- Expand the dataset to include more diverse conversational pairs.
- Experiment with other Transformer models, such as T5 or ALBERT.
- Improve the chatbot's ability to handle out-of-domain queries more gracefully.

---

This project provides a solid foundation for building and deploying domain-specific chatbots using state-of-the-art NLP techniques. The detailed documentation and demo video ensure that the project is reproducible and accessible to others interested in similar applications.
