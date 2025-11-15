from retriever import retrieve
from generator import generate_answer, check_special_input

print("\n🤖 PyFacBot RAG — Powered by Mistral")
print("-----------------------------------------")

while True:
    query = input("\n💬 Vous : ").strip()
    if not query:
        continue

    # Vérifie si c'est une salutation, merci ou au revoir
    special, lang = check_special_input(query)
    if special:
        print("\n🤖 :", special)
        if any(w in query.lower() for w in ["au revoir", "ciao", "bye"]):
            break
        continue

    # Récupération des passages pertinents
    print("\n🔍 Recherche dans les documents...")
    passages = retrieve(query)

    if not passages:
        print("\n🤖 : Aucune information pertinente trouvée.")
        continue

    # Génération de la réponse
    print("\n⚙️ Génération de la réponse...")
    answer = generate_answer(query, passages, lang)
    print("\n🤖 :", answer)
import re
from retriever import retrieve
from generation import generate_answer, check_special_input

print("\n🤖 PyFacBot RAG — Powered by Mistral")
print("-----------------------------------------")

while True:
    query = input("\n💬 Vous : ").strip()
    if not query:
        continue

    query_clean = re.sub(r"[^\w\s]", "", query)

    # Vérification salutations / au revoir / merci
    special, lang = check_special_input(query_clean)
    if special:
        print("\n🤖 :", special)
        if any(word in query_clean.lower() for word in ["au revoir", "ciao", "bye", "goodbye", "see you"]):
            break
        continue

    # Récupération des passages pertinents
    print("\n🔍 Recherche dans les documents...")
    passages = retrieve(query_clean)
    passages = [p for p in passages if p.get("score",0) > 0.1]

    if not passages:
        print("\n🤖 : Je n'ai pas trouvé de document suffisamment pertinent.")
        continue

    # Génération réponse
    print("\n⚙️ Génération de la réponse...")
    answer = generate_answer(query_clean, passages, lang)
    print("\n🤖 :", answer)
