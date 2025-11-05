import speech_recognition as sr
import pyttsx3
from retriever import retrieve
from generator import generate_answer
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import CompleteStyle

STOP_WORDS = ["stop", "stoppe", "stope", "terminé", "termine", "terminer"]


# === Synthèse vocale ===
def speak(text):
    """
    Fait parler le chatbot (voix naturelle locale).
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)  # vitesse
    voices = engine.getProperty("voices")
    # Essaie de trouver une voix française si possible
    for v in voices:
        if "fr" in v.languages or "French" in v.name:
            engine.setProperty("voice", v.id)
            break
    engine.say(text)
    engine.runAndWait()


# === Écoute continue jusqu'à un mot d'arrêt ===
def listen_until_stop():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("\n🎙️  Vous pouvez parler. (Dites 'terminé' ou 'stop' pour envoyer la requête)")
    print("🎧 En écoute...\n")

    full_text = ""
    try:
        while True:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("🎤 Parlez maintenant...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)

            try:
                text = recognizer.recognize_google(audio, language="fr-FR").strip()
                print(f"🗣️  {text}")
                if any(stop in text.lower() for stop in STOP_WORDS):
                    print("🛑 Arrêt de l'écoute.")
                    break
                full_text += " " + text
            except sr.UnknownValueError:
                print("🤔 (Je n’ai pas compris, continuez...)")
            except sr.RequestError:
                print("❌ Erreur de reconnaissance vocale.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Enregistrement interrompu manuellement.")
        return full_text.strip()

    return full_text.strip()


# === Correction inline ===
def correction_step(detected_text):
    print("\n✏️  Correction du texte :")
    print("(Modifiez directement le texte si nécessaire, puis appuyez sur Entrée)\n")

    corrected = prompt(
        f"👉 Corrigez ci-dessous : ",
        default=detected_text,  # prérempli
        complete_style=CompleteStyle.READLINE_LIKE
    ).strip()

    print(f"\n✅ Texte corrigé : {corrected}\n")
    return corrected


# === Boucle principale ===
def main():
    print("🤖 Chatbot RAG vocal avec Ollama (LLaMA 3)")
    print("--------------------------------------------------\n")

    while True:
        try:
            spoken_text = listen_until_stop()
            if not spoken_text:
                print("⚠️ Aucune entrée détectée.")
                continue

            corrected_text = correction_step(spoken_text)
            if corrected_text.lower() in ["exit", "quitter"]:
                print("👋 Fin du programme.")
                speak("Au revoir !")
                break

            print("📚 Recherche des passages pertinents...")
            passages = retrieve(corrected_text)
            
            print("\n💡 Génération de la réponse...\n")
            answer = generate_answer(corrected_text, passages)
            
            print("💬 Réponse du chatbot :")
            print(answer)
            print("\n" + "="*60 + "\n")

            # 🔊 Le chatbot lit la réponse à voix haute
            speak(answer)

        except KeyboardInterrupt:
            print("\n🛑 Programme arrêté manuellement.")
            break


if __name__ == "__main__":
    main()
