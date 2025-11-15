from flask import Flask, request, jsonify
from flask_cors import CORS
from retriever import retrieve
from generator import generate_answer, check_special_input
import pyttsx3
import threading
import re

app = Flask(__name__)
CORS(app)

# === Synthèse vocale locale ===
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    voices = engine.getProperty("voices")
    for v in voices:
        if "fr" in v.languages or "French" in v.name:
            engine.setProperty("voice", v.id)
            break
    engine.say(text)
    engine.runAndWait()

# === Endpoint chat ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "⚠️ Aucun texte reçu."})

    # Nettoyage du texte
    query_clean = re.sub(r"[^\w\s]", "", question)

    # Vérification salutations / merci / au revoir
    special, lang = check_special_input(query_clean)
    if special:
        threading.Thread(target=speak, args=(special,)).start()
        return jsonify({"answer": special})

    print("⏳ PyFacBot is thinking...", end="", flush=True)

    # Récupération des passages pertinents
    passages = retrieve(query_clean)

    if not passages:
        no_info = "Je n'ai pas trouvé de document suffisamment pertinent pour répondre." if lang=="fr" else "No relevant documents found."
        threading.Thread(target=speak, args=(no_info,)).start()
        return jsonify({"answer": no_info})

    # Génération de la réponse
    answer = generate_answer(query_clean, passages, lang)

    print("\r" + " " * 80 + "\r", end="", flush=True)
    print(f"[CHAT] Q: {question} | A: {answer}", flush=True)

    threading.Thread(target=speak, args=(answer,)).start()
    return jsonify({"answer": answer})

if __name__ == "__main__":
    print("🚀 Flask API running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
