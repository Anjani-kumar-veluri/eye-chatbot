from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from transformers import BertTokenizer, BertForSequenceClassification
from langchain_community.llms import CTransformers

from rapidfuzz import fuzz
from spellchecker import SpellChecker

import torch
import os


HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(HF_TOKEN)
    print("HF TOKEN loaded successfully")
else:
    print("HF TOKEN not found. If needed, set it using: export HF_TOKEN='your_token'")


DB_FAISS_PATH = r"database"
MODEL_PATH = r"bert_medical_classifier_pytorch/bert_medical_classifier_pytorch"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()


spell = SpellChecker()

CUSTOM_MEDICAL_WORDS = [
    "myopia", "hyperopia", "astigmatism", "glaucoma", "cataract",
    "conjunctivitis", "retina", "cornea", "lasik", "floaters",
    "blurred", "vision", "itchy", "retinal", "detachment",
    "trauma", "chemical", "ointment", "antibiotic", "ophthalmology",
    "viva", "keratitis", "uveitis", "blepharitis", "presbyopia",
    "amblyopia", "strabismus", "retinopathy", "macula", "macular",
    "photophobia", "diplopia", "ptosis", "nystagmus", "chalazion",
    "stye", "hordeolum", "pterygium", "pinguecula", "keratoconus",
    "iritis", "scleritis", "episcleritis", "endophthalmitis",
    "dacryocystitis", "trachoma"
]

spell.word_frequency.load_words(CUSTOM_MEDICAL_WORDS)


def correct_query(query):
    words = query.lower().split()
    corrected_words = []

    for word in words:
        clean_word = word.strip(".,!?;:'\"()[]{}")

        if len(clean_word) <= 3:
            corrected_words.append(clean_word)
            continue

        correction = spell.correction(clean_word)

        if correction and correction != clean_word:
            score = fuzz.ratio(clean_word, correction)

            if score >= 70:
                corrected_words.append(correction)
            else:
                corrected_words.append(clean_word)
        else:
            corrected_words.append(clean_word)

    return " ".join(corrected_words)


def fuzzy_match(query, keywords, threshold=70):
    query = query.lower()

    for keyword in keywords:
        score = fuzz.partial_ratio(query, keyword.lower())
        if score >= threshold:
            return True

    return False


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = get_embedding_model()


llm = CTransformers(
    model=r"models2/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    config={
        "max_new_tokens": 250,
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


follow_up_prompt_template = """
You are an ophthalmology chatbot helping users diagnose symptoms.
Given the user's symptom, generate the five most important follow-up questions, one per line.

User Symptom: {user_input}

Best 5 follow-up questions:
"""


diagnosis_prompt_template = """
You are an ophthalmology assistant. Based on the patient's symptoms and their responses to five key follow-up questions,
provide a concise, accurate diagnosis or recommendation.

Relevant document information:
{faiss_text}

User Symptom: {user_input}
User Responses:
{user_responses}

Final Diagnosis or Recommendation:
"""


treatment_prompt_template = """
You are an ophthalmology expert providing treatment recommendations.

Relevant document information:
{faiss_text}

Based on the user's symptom, provide the best possible treatment options, home remedies, or medications.

User Symptom: {user_input}

Best Treatment or Cure:
"""


GREETINGS = [
    "hello", "hi", "hey", "hi there", "hello there", "hey there",
    "good morning", "good afternoon", "good evening", "good day",
    "greetings", "welcome", "welcome back", "nice to meet you",
    "how are you", "how are you doing", "what's up", "good to see you",
    "howdy", "sup", "yo"
]


GOODBYES = [
    "bye", "goodbye", "exit", "quit", "see you", "bye bye",
    "see you soon", "see you later", "talk to you later",
    "take care", "have a nice day", "good night"
]


EYE_DISEASE_NAMES = [
    "myopia", "hyperopia", "astigmatism", "glaucoma", "cataract",
    "conjunctivitis", "retinal detachment", "diabetic retinopathy",
    "macular degeneration", "dry eye", "blepharitis", "keratitis",
    "uveitis", "presbyopia", "strabismus", "amblyopia",
    "night blindness", "color blindness", "corneal ulcer",
    "optic neuritis", "retinitis pigmentosa", "keratoconus",
    "chalazion", "stye", "hordeolum", "pterygium", "pinguecula",
    "scleritis", "episcleritis", "iritis", "endophthalmitis",
    "dacryocystitis", "trachoma"
]


EYE_EMERGENCY_KEYWORDS = [
    "sudden vision loss", "eye injury", "chemical in eye",
    "severe eye pain", "flashes of light", "retinal detachment",
    "eye trauma", "vision blackout", "foreign object in eye",
    "something went into my eye", "object in eye", "eye bleeding",
    "sudden blindness", "cannot see suddenly"
]


EYE_TREATMENT_KEYWORDS = [
    "eye drops", "treatment", "cure", "medicine", "laser surgery",
    "lasik", "cataract surgery", "how to treat", "remedy",
    "eye care", "how to cure", "treat my", "cure my", "solution"
]


EYE_DIAGNOSIS_KEYWORDS = [
    "blurred vision", "itchy eyes", "red eyes", "dry eyes",
    "eye pain", "double vision", "night blindness", "floaters",
    "halos", "vision problem", "i have eye", "my eye hurts",
    "eye irritation", "watering eyes", "burning eyes",
    "cannot see clearly", "blur vision"
]


EYE_GENERAL_KEYWORDS = [
    "what is", "explain", "meaning of", "causes of", "symptoms of",
    "eye disease", "eye condition", "tell me about"
]


EYE_MEDICINE_KEYWORDS = [
    "side effects", "uses of eye drops", "dosage", "ointment",
    "antibiotic eye drops", "side effects of medicine",
    "how much eye drops", "eye medicine dosage"
]


EYE_HEALTH_ADVICE_KEYWORDS = [
    "can i use screen", "eye strain", "how to protect eyes",
    "tips for eye care", "is screen harmful",
    "how to reduce eye strain", "blue light effect",
    "screen time", "protect my eyes", "healthy eyes"
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
    query = correct_query(user_input.lower())
    return fuzzy_match(query, OPHTHAL_VIVA_KEYWORDS, threshold=70)


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

    remove_words = [
        "i completed", "test me on", "test me", "quiz me on",
        "ask questions on", "ophthal viva", "ophthalmology viva",
        "viva", "quiz me", "ask questions"
    ]

    for word in remove_words:
        topic = topic.replace(word, "")

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


def search_faiss_db(query, k=1):
    results = db.similarity_search(query, k=k)
    return results


def handle_greetings_and_goodbyes(user_input):
    user_input = user_input.lower().strip()
    words = user_input.split()

    if user_input in GREETINGS:
        return "Hello! How can I assist you with your eye-related medical concerns today?"

    if any(word in GOODBYES for word in words):
        return "Take care! Stay healthy."

    return None


def generate_best_questions(user_input):
    prompt = PromptTemplate(
        template=follow_up_prompt_template,
        input_variables=["user_input"]
    )

    question_chain = LLMChain(llm=llm, prompt=prompt)

    return question_chain.invoke({
        "user_input": user_input
    })["text"].strip().split("\n")


def generate_final_diagnosis(user_input, user_responses):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])

    prompt = PromptTemplate(
        template=diagnosis_prompt_template,
        input_variables=["user_input", "user_responses", "faiss_text"]
    )

    diagnosis_chain = LLMChain(llm=llm, prompt=prompt)

    return diagnosis_chain.invoke({
        "user_input": user_input,
        "user_responses": user_responses,
        "faiss_text": faiss_text
    })["text"].strip()


def generate_treatment(user_input):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])

    prompt = PromptTemplate(
        template=treatment_prompt_template,
        input_variables=["user_input", "faiss_text"]
    )

    treatment_chain = LLMChain(llm=llm, prompt=prompt)

    return treatment_chain.invoke({
        "user_input": user_input,
        "faiss_text": faiss_text
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


def faiss_has_eye_context(query):
    try:
        results = search_faiss_db(query, k=1)

        if not results:
            return False

        text = results[0].page_content.lower()

        eye_terms = [
            "eye", "vision", "retina", "cornea", "lens", "optic",
            "glaucoma", "cataract", "myopia", "hyperopia",
            "astigmatism", "conjunctivitis", "ophthalmology"
        ]

        return any(term in text for term in eye_terms)

    except Exception:
        return False


def generate_general_medical_info(user_input):
    faiss_results = search_faiss_db(user_input)
    faiss_text = "\n".join([result.page_content for result in faiss_results])

    prompt = PromptTemplate(
        template=(
            "Explain the following ophthalmology condition based on the relevant documents:\n"
            "{faiss_text}\n\n"
            "User input: {user_input}\n\n"
            "Provide a clear explanation with:\n"
            "1. What it is\n"
            "2. Common symptoms\n"
            "3. Common causes\n"
            "4. Basic treatment or management\n"
            "5. When to consult an eye doctor\n"
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
            "Provide detailed information about the following eye medicine based on the documents:\n"
            "{faiss_text}\n\n"
            "User input: {user_input}\n\n"
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
            "Give eye health advice based on the following user query:\n"
            "{user_input}\n\n"
            "Relevant information from documents:\n"
            "{faiss_text}\n\n"
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
            "Provide emergency eye advice based on the user query:\n"
            "{user_input}\n\n"
            "Relevant emergency information from documents:\n"
            "{faiss_text}\n\n"
            "Provide clear and actionable emergency steps. Also advise the user to consult an eye doctor immediately if serious symptoms are present."
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
    corrected_query = correct_query(query.lower())

    if is_opthal_viva_query(corrected_query):
        return "opthal_viva"

    if fuzzy_match(corrected_query, EYE_EMERGENCY_KEYWORDS, threshold=65):
        return "emergency_advice"

    elif fuzzy_match(corrected_query, EYE_TREATMENT_KEYWORDS, threshold=65):
        return "treatment"

    elif fuzzy_match(corrected_query, EYE_MEDICINE_KEYWORDS, threshold=65):
        return "medicine_info"

    elif fuzzy_match(corrected_query, EYE_HEALTH_ADVICE_KEYWORDS, threshold=65):
        return "health_advice"

    elif fuzzy_match(corrected_query, EYE_DIAGNOSIS_KEYWORDS, threshold=65):
        return "diagnosis"

    elif fuzzy_match(corrected_query, EYE_DISEASE_NAMES, threshold=75):
        return "general_medical"

    elif fuzzy_match(corrected_query, EYE_GENERAL_KEYWORDS, threshold=65):
        return "general_medical"

    else:
        if classify_query(corrected_query):
            return "diagnosis"

        if faiss_has_eye_context(corrected_query):
            return "general_medical"

        return "general_medical"