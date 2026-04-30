# from langchain_huggingface import HuggingFaceEndpoint

from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
# from huggingface_hub import login

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from transformers import BertTokenizer, BertForSequenceClassification
import torch

DB_FAISS_PATH = r"/home/bhcp0089/Desktop/AiMedicalChatbot_updated/database"

MODEL_PATH = r"/home/bhcp0089/Desktop/AiMedicalChatbot_updated/bert_medical_classifier_pytorch/bert_medical_classifier_pytorch"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


embedding_model = get_embedding_model()

# ✅ Load LLM only once globally
llm = CTransformers(
    model=r"/home/bhcp0089/Desktop/AiMedicalChatbot_updated/models2/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    config={
        "max_new_tokens": 100,
        "temperature": 0.3,
        "threads": 4,
        "context_length": 1024
    }
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


def handle_greetings_and_goodbyes(user_input):
    user_input = user_input.lower().strip()

    GREETINGS = [
        "hello", "hi", "hey", "hi there", "hello there", "hey there",
        "good morning", "good afternoon", "good evening", "good day",
        "greetings", "warm greetings", "season's greetings", "compliments of the day",
        "welcome", "welcome back", "a warm welcome",
        "nice to meet you", "pleased to meet you", "it's a pleasure to meet you",
        "how are you", "how are you doing", "how's it going", "how have you been",
        "what's up", "what's new", "long time no see",
        "good to see you", "great to see you",
        "thanks for reaching out", "thank you for contacting us",
        "good to connect with you", "hello everyone", "hi everyone"
    ]

    GOODBYES = [
        "bye", "goodbye", "exit", "quit", "see you", "bye bye", "see you soon",
        "see you later", "see you tomorrow", "talk to you later",
        "catch you later", "farewell", "take care",
        "have a nice day", "have a great day", "have a good day",
        "have a wonderful day", "have a good evening",
        "good night", "all the best", "best wishes",
        "until next time", "keep in touch",
        "thanks, goodbye", "thank you, bye",
        "it was nice talking to you", "nice chatting with you"
    ]

    if user_input in GREETINGS:
        return "Hello! How can I assist you with your medical concerns today?"
    elif user_input in GOODBYES:
        return "Take care! Stay healthy."

    return None


follow_up_prompt_template = """
You are a medical chatbot helping users diagnose symptoms.
Given the user's symptom, generate the **five most important** follow-up questions, one per line.

User Symptom: {user_input}

Best 5 follow-up questions:
"""

diagnosis_prompt_template = """
You are a medical assistant. Based on the patient's symptoms and their responses to five key follow-up questions,
provide a concise, accurate diagnosis or recommendation.

User Symptom: {user_input}

User Responses:
{user_responses}

Final Diagnosis or Recommendation:
"""

treatment_prompt_template = """
You are a medical expert providing treatment recommendations.
Based on the user's symptom, provide the best possible treatment options, home remedies, or medications.

User Symptom: {user_input}

Best Treatment or Cure:
"""


def generate_best_questions(user_input, conversation_history):
    prompt = PromptTemplate(
        template=follow_up_prompt_template,
        input_variables=["user_input", "conversation_history"]
    )

    question_chain = LLMChain(llm=llm, prompt=prompt)

    return question_chain.invoke({
        "user_input": user_input,
        "conversation_history": conversation_history
    })["text"].strip().split("\n")


def generate_final_diagnosis(user_input, user_responses):
    prompt = PromptTemplate(
        template=diagnosis_prompt_template,
        input_variables=["user_input", "user_responses"]
    )

    diagnosis_chain = LLMChain(llm=llm, prompt=prompt)

    return diagnosis_chain.invoke({
        "user_input": user_input,
        "user_responses": user_responses
    })["text"].strip()


def generate_treatment(user_input):
    prompt = PromptTemplate(
        template=treatment_prompt_template,
        input_variables=["user_input"]
    )

    treatment_chain = LLMChain(llm=llm, prompt=prompt)

    return treatment_chain.invoke({
        "user_input": user_input
    })["text"].strip()


def classify_query(query):
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return predicted_class == 1


if __name__ == "__main__":
    while True:
        user_query = input("Write your symptoms or query: ").strip().lower()

        response = handle_greetings_and_goodbyes(user_query)

        if response:
            print("\nBot:", response)

            if user_query in [
                "bye", "goodbye", "exit", "quit", "see you", "bye bye", "see you soon",
                "see you later", "see you tomorrow", "talk to you later",
                "catch you later", "farewell", "take care",
                "have a nice day", "have a great day", "have a good day",
                "have a wonderful day", "have a good evening",
                "good night", "all the best", "best wishes",
                "until next time", "keep in touch",
                "thanks, goodbye", "thank you, bye",
                "it was nice talking to you", "nice chatting with you"
            ]:
                break

            continue

        if not classify_query(user_query):
            print("\nBot: I'm a medical assistant and can only help with health-related questions.")
            continue

        if any(keyword in user_query.lower() for keyword in ["cure", "treatment", "medicine", "remedy"]):
            treatment = generate_treatment(user_query)
            print("\nBot: Here is the recommended treatment:\n", treatment)
            continue

        follow_up_questions = generate_best_questions(user_query, "")
        user_responses = []

        for i, question in enumerate(follow_up_questions, start=1):
            user_answer = input(f"Bot ({i}/5): {question}\nYour response: ")
            user_responses.append(f"Q{i}: {question} | A{i}: {user_answer}")

        combined_responses = "\n".join(user_responses)

        final_diagnosis = generate_final_diagnosis(user_query, combined_responses)

        print("\nBot: Here is your final diagnosis:\n", final_diagnosis)