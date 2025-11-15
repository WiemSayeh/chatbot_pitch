import ollama
import re
from langdetect import detect

MAX_CONTEXT_CHARS = 4000

# ===== Fonctions utilitaires =====
def clean_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def sanitize_text(text):
    text = text.replace("*", "").replace("+", "")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def structure_response(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    structured = []
    seen = set()
    for s in sentences:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        structured.append(f"• {s}" if len(s) > 50 else s)
    return "\n".join(structured)

def truncate_context(passages):
    context = ""
    for p in passages:
        text = sanitize_text(clean_text(p['text']))
        context += "\n" + text
    return context.strip()

# ===== Gestion salutations / remerciements / au revoir =====
SALUTATIONS = ["bonjour", "salut", "coucou", "hello", "hi", "hey"]
AUREVOIRS = ["au revoir", "à bientôt", "ciao", "bye", "goodbye", "see you"]
THANKS = ["merci", "merci beaucoup", "thanks", "thank you", "thx"]

def check_special_input(query):
    lang = "fr"
    try:
        lang_detected = detect(query)
        if lang_detected in ["en", "fr"]:
            lang = lang_detected
    except:
        pass
    q_lower = query.lower()
    if any(word in q_lower for word in SALUTATIONS):
        return ("Hello! 👋 I am PyFacBot, ready to assist you!" if lang=="en"
                else "Bonjour ! 👋 Je suis PyFacBot, ravi de vous aider !"), lang
    if any(word in q_lower for word in THANKS):
        return ("You're welcome! 😊" if lang=="en"
                else "Je vous en prie ! 😊"), lang
    if any(word in q_lower for word in AUREVOIRS):
        return ("Goodbye! 👋 See you soon." if lang=="en" 
                else "Au revoir ! 👋 À bientôt."), lang   
    return None, lang

# ===== Génération réponse RAG =====
def generate_answer(query, passages, lang="fr"):
    if not passages:
        return "Désolé, je n'ai pas d'information sur ce sujet." if lang=="fr" else "Sorry, I don't have information on that."

    context_text = truncate_context(passages)

    if lang=="en":
        system_prompt = "You are PyFacBot, official chatbot. Answer clearly and concisely using bullet points."
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer concisely with bullet points."
    else:
        system_prompt = "Tu es PyFacBot, le chatbot officiel. Réponds clairement et de manière concise en utilisant des tirets."
        user_prompt = f"Contexte:\n{context_text}\n\nQuestion: {query}\nFournis une réponse concise et structurée avec des points clés."

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}]
        )
    except Exception as e:
        return f"Erreur lors de la génération : {e}"

    text = getattr(response, "message", None)
    text = text.content if text else str(response)
    return structure_response(clean_text(text))
