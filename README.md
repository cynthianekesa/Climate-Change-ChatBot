# Climate-Change-ChatBot
This project implements a Climate Change Chatbot using a pre-trained language model fine-tuned on a climate change-related dataset. The chatbot is designed to answer questions and provide accurate information about climate change, its impacts, and potential solutions for improved climate change awareness and empowerment of individuals in its fight.

#### 1. **Introduction**
This report outlines the development of a **Climate Change Chatbot** using a pre-trained Transformer model from Hugging Face. The chatbot is designed to answer questions related to climate change, providing accurate and contextually relevant responses. The project leverages Hugging Face's `transformers` library and TensorFlow for fine-tuning and deployment. The chatbot is domain-specific, focusing on climate change, and is evaluated using both quantitative metrics and qualitative testing.

---

#### 2. **Dataset Collection and Preprocessing**
The dataset used for training the chatbot consists of **conversational pairs** related to climate change. The dataset was either collected from publicly available sources or created manually to ensure diversity and coverage of various user intents. The dataset includes questions and answers on topics such as global warming, renewable energy, carbon emissions, and climate policies.

**Preprocessing Steps:**
- **Tokenization**: The text data was tokenized using the tokenizer associated with the pre-trained model (e.g., BERT or GPT-2 tokenizer).
- **Normalization**: Text was normalized by converting to lowercase, removing special characters, and handling contractions.
- **Handling Missing Values**: Any incomplete or missing conversational pairs were either removed or manually corrected to ensure data quality.
- **Dataset Splitting**: The dataset was split into training, validation, and test sets (e.g., 80% training, 10% validation, 10% testing).

The preprocessed dataset was saved in a format compatible with Hugging Face's `datasets` library, such as JSON or CSV.

---

#### 3. **Model Selection and Fine-Tuning**
A pre-trained Transformer model was selected from Hugging Face's model hub. For this project, **BERT** or **GPT-2** was chosen due to their strong performance in natural language understanding and generation tasks.

**Fine-Tuning Process:**
- **Model Initialization**: The pre-trained model was loaded using Hugging Face's `transformers` library.
- **Hyperparameter Tuning**: Key hyperparameters were tuned, including:
  - **Learning Rate**: Tested values between 1e-5 and 5e-5.
  - **Batch Size**: Experimented with batch sizes of 16, 32, and 64.
  - **Optimizer**: AdamW optimizer was used with weight decay.
  - **Training Epochs**: Trained for 3–5 epochs to avoid overfitting.
- **Training**: The model was fine-tuned on the climate change dataset using TensorFlow. Training progress was monitored using validation loss and accuracy.

**Impact of Hyperparameter Adjustments:**
- A lower learning rate (e.g., 2e-5) resulted in more stable training and better convergence.
- A batch size of 32 provided a good balance between training speed and memory usage.
- Training for 4 epochs yielded the best performance without overfitting.

---

#### 4. **Evaluation**
The chatbot was evaluated using both quantitative metrics and qualitative testing.

**Quantitative Metrics:**
- **BLEU Score**: Used to evaluate the quality of generated responses by comparing them to reference answers.
- **F1-Score**: Measured the chatbot's ability to correctly classify and respond to user intents.
- **Perplexity**: Assessed the model's confidence in generating responses.

**Qualitative Testing:**
- The chatbot was tested with a variety of climate change-related questions, such as:
  - "What are the main causes of global warming?"
  - "How can we reduce carbon emissions?"
- The chatbot's responses were analyzed for relevance, accuracy, and coherence. It was also tested with out-of-domain queries to ensure it could reject or handle them appropriately.

---

#### 5. **Deployment**
The chatbot was deployed using **Gradio**, a simple and intuitive web interface framework. The deployment process included:
- **Web Interface**: Users can input their questions in a text box and receive responses from the chatbot in real-time.
- **API Integration**: The chatbot was also deployed as an API using Flask, allowing integration with other applications.

**User Interaction:**
- The interface is user-friendly, with clear instructions for inputting queries.
- Example conversations were provided to guide users on how to interact with the chatbot.

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
