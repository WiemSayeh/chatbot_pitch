import speech_recognition as sr
import pyttsx3
from retriever import retrieve
from generator import generate_answer

# === Synthèse vocale ===
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[0].id)
    engine.say(text)
    engine.runAndWait()

# === Reconnaissance vocale longue ===
def listen_long():
    """
    Écoute la voix de l’utilisateur sans couper trop vite.
    L’écoute se termine seulement si :
    - l’utilisateur dit 'stop', 'terminé', ou 'quitter'
    - un silence très long est détecté
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎙️  Vous pouvez parler. (Dites 'terminé' ou 'stop' pour envoyer la requête)")
        r.adjust_for_ambient_noise(source, duration=1)
        audio_data = []
        silent_count = 0
        print("🎧 En écoute...")

        while True:
            try:
                # écoute par segments courts (phrase par phrase)
                audio = r.listen(source, phrase_time_limit=10, timeout=None)
                text = r.recognize_google(audio, language="fr-FR").lower()
                print(f"🗣️  {text}")

                # si l'utilisateur dit "terminé" → fin
                if any(stop_word in text for stop_word in ["terminé", "stop", "quitter", "envoyer"]):
                    print("✅ Fin de la prise de parole.")
                    break

                audio_data.append(text)
                silent_count = 0

            except sr.UnknownValueError:
                silent_count += 1
                if silent_count >= 3:
                    print("🤫 Silence prolongé détecté, arrêt de l'écoute.")
                    break
            except KeyboardInterrupt:
                print("\n🛑 Arrêt manuel.")
                break

    # joindre toutes les phrases détectées
    final_text = " ".join(audio_data).strip()
    if not final_text:
        print("❌ Aucun texte détecté.")
        return None
    return final_text

# === Correction du texte reconnu ===
def edit_text(initial_text):
    print("\n✏️  Correction du texte :")
    print("(Appuyez sur Entrée sans rien écrire pour valider la version actuelle)\n")
    print("👉 Corrigez ci-dessous et appuyez sur Entrée : ", end="")
    user_edit = input(initial_text).strip()
    return user_edit if user_edit else initial_text

# === Boucle principale ===
def main():
    print("🤖 Chatbot RAG vocal avec Ollama (LLaMA 3)")
    print("--------------------------------------------------")

    while True:
        # 🎧 Écoute prolongée
        query = listen_long()
        if not query:
            continue

        # 📴 Quitter
        if query.lower() in ["terminé", "exit", "quitter", "stop"]:
            print("\n👋 Fin du programme.")
            speak("Au revoir !")
            break

        # ✏️ Correction légère avant envoi
        query = edit_text(query)

        # 🔍 Récupération des passages
        print("\n📚 Recherche des passages pertinents...")
        passages = retrieve(query)

        # 💡 Génération de la réponse
        print("\n💡 Génération de la réponse...\n")
        answer = generate_answer(query, passages)

        # 🗣️ Affichage + synthèse vocale
        print("💬 Réponse du chatbot :")
        print(answer)
        print("\n" + "="*60 + "\n")
        speak(answer)

if __name__ == "__main__":
    main()
