from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
import pandas as pd
import os
from transformers import pipeline

# Load the conversational pipeline from the transformers library
chat_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Initialize memory storage
memory = {}

# Create a new chatbot instance
chatbot = ChatBot(
    'EduBot', 
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///db.sqlite3',  # Explicit database location
    logic_adapters=['chatterbot.logic.BestMatch']
)

# Create trainers
corpus_trainer = ChatterBotCorpusTrainer(chatbot)
custom_trainer = ListTrainer(chatbot)

# Train the chatbot using the English corpus
corpus_trainer.train(
    "chatterbot.corpus.english",
    "chatterbot.corpus.english.greetings",
    "chatterbot.corpus.english.conversations"
)

# Load custom dataset
csv_path = r"educational\Dataset_Python_Question_Answer.csv"

if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)

    # Drop any row that has missing values (NaN)
    data = data.dropna()

    # Convert all values to strings (to avoid float issues)
    data = data.astype(str)

    # Train the bot with Q&A pairs
    questions_answers = list(zip(data.iloc[:, 0], data.iloc[:, 1]))  # Assuming 2-column CSV: Q & A
    for question, answer in questions_answers:
        custom_trainer.train([question.strip(), answer.strip()])  # Strip extra spaces
else:
    print(f"Warning: CSV file '{csv_path}' not found! Skipping custom training. ❌")

# Function to get a response with memory integration
def chat_with_memory(user_input):
    # Check if the user mentions their name
    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is ")[1]
        memory['name'] = name.strip()
        return f"Nice to meet you, {memory['name']}! 😊"
    
    # Check if the user has introduced themselves already
    if 'name' in memory:
        user_input = f"Hi, my name is {memory['name']}. {user_input}"
    
    # Respond using the ChatterBot or DialoGPT pipeline (for more natural conversations)
    response = chatbot.get_response(user_input)  # ChatterBot's response
    if response.confidence < 0.5:  # If ChatterBot is uncertain, use DialoGPT
        response = chat_pipeline(user_input, max_length=1000, pad_token_id=50256)
        response = response[0]['generated_text']
    
    return response

# Main chat loop
print("Hello! I'm EduBot🤖. Ask me educational questions.")
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Have a great day!👋")
            break
        response = chat_with_memory(user_input)
        print(f"EduBot🤖: {response}")
    except KeyboardInterrupt:
        print("\nGoodbye!👋 Exiting...")
        break