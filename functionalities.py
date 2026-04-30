from langchain_huggingface import HuggingFaceEndpoint # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from huggingface_hub import login
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

from langchain_community.llms import CTransformers

HF_TOKEN = os.getenv("HF_TOKEN")

login(HF_TOKEN)
print("HF TOKEN:", HF_TOKEN)

huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
DB_FAISS_PATH = r"/home/bhcp0089/Desktop/AiMedicalChatbot_updated/database"

MODEL_PATH = r"/home/bhcp0089/Desktop/AiMedicalChatbot_updated/bert_medical_classifier_pytorch/bert_medical_classifier_pytorch"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

def get_embedding_model():
  embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  return embedding_model

embedding_model = get_embedding_model()

# ✅ Load LLM only once globally
llm = CTransformers(
    model=r"/home/bhcp0089/Desktop/AiMedicalChatbot_updated/models2/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    config={
        "max_new_tokens": 250,
        "temperature": 0.3,
        "threads": 4,
        "context_length": 1024
    }
)

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)



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
    "howdy", "sup", "yo"
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

EYE_EMERGENCY_KEYWORDS = [
    "sudden vision loss", "eye injury", "chemical in eye",
    "severe eye pain", "flashes of light", "retinal detachment",
    "eye trauma", "vision blackout", "foreign object in eye"
]

EYE_TREATMENT_KEYWORDS = [
    "eye drops", "treatment for", "cure for", "medicine for",
    "laser surgery", "lasik", "cataract surgery",
    "how to treat", "remedy for", "eye care"
]

EYE_DIAGNOSIS_KEYWORDS = [
    "blurred vision", "itchy eyes", "red eyes", "dry eyes",
    "eye pain", "double vision", "night blindness",
    "floaters", "halos", "vision problem", "i have eye"
]

EYE_GENERAL_KEYWORDS = [
    "what is cataract", "what is glaucoma", "what is myopia",
    "what is hyperopia", "astigmatism", "eye disease",
    "causes of eye", "symptoms of eye", "eye condition"
]

EYE_MEDICINE_KEYWORDS = [
    "eye drops side effects", "uses of eye drops",
    "dosage of eye medicine", "ointment for eyes",
    "antibiotic eye drops"
]

EYE_HEALTH_ADVICE_KEYWORDS = [
    "can i use screen", "eye strain", "how to protect eyes",
    "tips for eye care", "is screen harmful",
    "how to reduce eye strain", "blue light effect"
]



OPHTHAL_VIVA_KEYWORDS = [
    "viva", "test me", "quiz me", "ask questions",
    "exam preparation", "i completed", "i studied",
    "ophthal viva", "ophthalmology viva"
]

viva_sessions = {}

viva_prompt_template = """
You are an ophthalmology viva examiner.

Topic: {topic}

Ask ONE short viva question strictly related to this topic.
Do NOT give the answer.
Do NOT explain.

Return only:
Question: <question>
"""

viva_answer_check_template = """
You are an ophthalmology viva examiner.

Topic: {topic}
Previous Question: {question}
Student Answer: {answer}

Evaluate the student's answer.
Give the correct answer for Previous Question.
Ask ONE next viva question from same topic.

Rules:
Do NOT answer the next question.
Do NOT use ---- or -- symbols.
Do NOT write extra explanation.

Return exactly:

Evaluation: Correct / Partially Correct / Incorrect

Correct Answer:
- point 1
- point 2
- point 3

Next Question: <one new viva question only>
"""


def is_opthal_viva_query(user_input):
    query = user_input.lower()
    return any(keyword in query for keyword in OPHTHAL_VIVA_KEYWORDS)


def clean_question(text):
    text = text.strip()

    if "Question:" in text:
        text = text.split("Question:", 1)[1].strip()

    if "Answer:" in text:
        text = text.split("Answer:", 1)[0].strip()

    if "Correct Answer:" in text:
        text = text.split("Correct Answer:", 1)[0].strip()

    lines = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
    return lines[0] if lines else text


def generate_next_question(topic):
    prompt = PromptTemplate(
        template=viva_prompt_template,
        input_variables=["topic"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    raw_q = chain.invoke({"topic": topic})["text"].strip()
    return clean_question(raw_q)


def generate_correct_answer(question):
    prompt = PromptTemplate(
        template="""
You are an ophthalmology expert.

Question: {question}

Give correct answer in 2-3 short bullet points.
Do NOT give generic answer.
Do NOT say review the concept.

Correct Answer:
""",
        input_variables=["question"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"question": question})["text"].strip()

    points = []
    for line in result.split("\n"):
        line = line.strip()
        if line and not line.lower().startswith("correct answer"):
            points.append(line.strip("- ").strip())

    return points[:3] if points else ["Correct answer could not be generated."]


def format_viva_response(raw_result, previous_question, topic):
    raw_result = raw_result.replace("----", "").replace("--", "").strip()

    evaluation = "Partially Correct"
    correct_answer = []
    next_question = ""

    mode = None

    for line in raw_result.split("\n"):
        line = line.strip()
        if not line:
            continue

        lower = line.lower()

        if lower.startswith("evaluation"):
            evaluation = line.split(":", 1)[-1].strip()
            mode = None

        elif lower.startswith("correct answer"):
            mode = "answer"

        elif lower.startswith("next question"):
            next_question = line.split(":", 1)[-1].strip()
            mode = None

        elif mode == "answer":
            if not lower.startswith("question") and not lower.startswith("answer"):
                correct_answer.append(line.strip("- ").strip())

    if not correct_answer:
        correct_answer = generate_correct_answer(previous_question)

    if not next_question:
        next_question = generate_next_question(topic)

    next_question = clean_question(next_question)

    formatted = f"Evaluation: {evaluation}\n\n"
    formatted += "Correct Answer:\n"

    for point in correct_answer[:3]:
        formatted += f"- {point}\n"

    formatted += f"\nNext Question: {next_question}"

    return formatted.strip(), next_question


def start_opthal_viva(user_id, user_input):
    topic = user_input.lower()

    topic = topic.replace("i completed", "")
    topic = topic.replace("test me on", "")
    topic = topic.replace("test me", "")
    topic = topic.replace("quiz me on", "")
    topic = topic.replace("ask questions on", "")
    topic = topic.replace("ophthal viva", "")
    topic = topic.replace("ophthalmology viva", "")
    topic = topic.strip()

    if not topic:
        topic = "ophthalmology"

    question = generate_next_question(topic)

    viva_sessions[user_id] = {
        "topic": topic,
        "last_question": question,
        "active": True
    }

    return "Ophthal Viva Started.\nTopic: " + topic + "\n\nQuestion: " + question


def continue_opthal_viva(user_id, user_answer):
    session = viva_sessions.get(user_id)

    if not session or not session["active"]:
        return "No viva session is active. Say: 'I completed cataract, test me.'"

    prompt = PromptTemplate(
        template=viva_answer_check_template,
        input_variables=["topic", "question", "answer"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    raw_result = chain.invoke({
        "topic": session["topic"],
        "question": session["last_question"],
        "answer": user_answer
    })["text"].strip()

    formatted_result, next_q = format_viva_response(
        raw_result,
        session["last_question"],
        session["topic"]
    )

    session["last_question"] = next_q
    viva_sessions[user_id] = session

    return formatted_result


def stop_opthal_viva(user_id):
    if user_id in viva_sessions:
        viva_sessions[user_id]["active"] = False
    return "Ophthal Viva stopped."


def search_faiss_db(query):
    results = db.similarity_search(query, k=1)
    return results

def handle_greetings_and_goodbyes(user_input):
    user_input = user_input.lower().strip()
    words = user_input.split()

    if user_input in GREETINGS:
        return "Hello! How can I assist you with your medical concerns today?"

    if any(word in GOODBYES for word in words):
        return "Take care! Stay healthy."

    return None

def generate_best_questions(user_input):
    prompt = PromptTemplate(template=follow_up_prompt_template, input_variables=["user_input"])
    question_chain = LLMChain(llm=llm, prompt=prompt)
    return question_chain.invoke({"user_input": user_input})["text"].strip().split("\n")

def generate_final_diagnosis(user_input, user_responses):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])
    prompt = PromptTemplate(template=diagnosis_prompt_template, input_variables=["user_input", "user_responses", "faiss_text"])
    diagnosis_chain = LLMChain(llm=llm, prompt=prompt)
    return diagnosis_chain.invoke({"user_input": user_input, "user_responses": user_responses, "faiss_text": faiss_text})["text"].strip()

def generate_treatment(user_input):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])
    prompt = PromptTemplate(template=treatment_prompt_template, input_variables=["user_input", "faiss_text"])
    treatment_chain = LLMChain(llm=llm, prompt=prompt)
    return treatment_chain.invoke({"user_input": user_input, "faiss_text": faiss_text})["text"].strip()

def classify_query(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class == 1

def generate_general_medical_info(user_input):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])

    prompt = PromptTemplate(
        template=(
            "Explain the following medical condition based on the relevant documents: '{faiss_text}'\n"
            "User input: {user_input}\n"
            "Provide a clear and concise explanation of the medical condition."
        ),
        input_variables=["user_input", "faiss_text"]
    )

    info_chain = LLMChain(llm=llm, prompt=prompt)

    result = info_chain.invoke({
        "user_input": user_input,
        "faiss_text": faiss_text
    })

    return result["text"].strip()

def generate_medicine_info(user_input):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])

    prompt = PromptTemplate(
        template=(
            "Provide detailed information about the following medicine based on the documents: '{faiss_text}'\n"
            "User input: {user_input}\n"
            "Provide clear and structured information about the medicine."
        ),
        input_variables=["user_input", "faiss_text"]
    )

    med_chain = LLMChain(llm=llm, prompt=prompt)

    result = med_chain.invoke({
        "user_input": user_input,
        "faiss_text": faiss_text
    })

    return result["text"].strip()

def generate_health_advice(user_input):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])

    prompt = PromptTemplate(
        template=(
            "Give health advice based on the following user query: '{user_input}'\n"
            "And the relevant information from these documents: '{faiss_text}'\n"
            "Provide clear, actionable health advice."
        ),
        input_variables=["user_input", "faiss_text"]
    )

    advice_chain = LLMChain(llm=llm, prompt=prompt)

    result = advice_chain.invoke({
        "user_input": user_input,
        "faiss_text": faiss_text
    })

    return result["text"].strip()

def generate_emergency_advice(user_input):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])

    prompt = PromptTemplate(
        template=(
            "Provide emergency advice based on the user query: '{user_input}'\n"
            "And the relevant emergency information from documents: '{faiss_text}'\n"
            "Provide clear and actionable emergency steps or advice."
        ),
        input_variables=["user_input", "faiss_text"]
    )

    emergency_chain = LLMChain(llm=llm, prompt=prompt)

    result = emergency_chain.invoke({
        "user_input": user_input,
        "faiss_text": faiss_text
    })

    return result["text"].strip()

def classify_query_type(query):
    query = query.lower()

    if is_opthal_viva_query(query):
        return "opthal_viva"

    if any(keyword in query for keyword in EYE_EMERGENCY_KEYWORDS):
        return "emergency_advice"

    elif any(keyword in query for keyword in EYE_TREATMENT_KEYWORDS):
        return "treatment"

    elif any(keyword in query for keyword in EYE_DIAGNOSIS_KEYWORDS):
        return "diagnosis"

    elif any(keyword in query for keyword in EYE_GENERAL_KEYWORDS):
        return "general_medical"

    elif any(keyword in query for keyword in EYE_MEDICINE_KEYWORDS):
        return "medicine_info"

    elif any(keyword in query for keyword in EYE_HEALTH_ADVICE_KEYWORDS):
        return "health_advice"

    else:
        return "unknown"