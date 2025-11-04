from retriever import retrieve
from generator import generate_answer

def main():
    print("💬 Chatbot RAG avec Ollama LLaMA 3\n")
    
    while True:
        try:
            query = input("Votre question (ou 'exit' pour quitter) : ")
        except EOFError:
            print("\nFin du programme.")
            break

        if query.lower() == "exit":
            print("\nFin du programme.")
            break

        # Recherche automatique dans tous les PDFs
        passages = retrieve(query)

        print("\n✅ Passages pertinents récupérés. Génération de la réponse...\n")
        answer = generate_answer(query, passages)
        
        # Affichage propre
        print("💬 Réponse du chatbot :")
        print(answer)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
