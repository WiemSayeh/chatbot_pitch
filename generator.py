import ollama

MODEL_NAME = "llama3"

def clean_text(text):
    """Nettoyage du texte pour supprimer répétitions et espaces inutiles."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def generate_answer(query, passages):
    """
    Génère une réponse claire et professionnelle à partir des passages pertinents.
    - Si la question concerne PyFac ou l'identité du chatbot, la réponse vient du PDF 'pyfac_info.pdf'.
    - Sinon, la réponse est générée à partir des autres contextes RAG.
    """

    context_text = "\n\n".join([p.get("text", "") for p in passages])

    # 🎯 Système de rôle du chatbot
    system_prompt = """
Tu es PyFacBot, le chatbot officiel de l’événement PyFac 11.
Tu as été développé par les étudiants du Département de Génie Informatique de l’ENIS.
Ta mission est de répondre aux questions liées :
- aux entreprises partenaires (Telnet, Sofrecom, KPIT, etc.)
- aux sujets technologiques et industriels présents dans les PDFs fournis.

PyFac 11 est un événement annuel du département de Génie Informatique
qui relie le monde académique et industriel à travers des conférences,
ateliers et présentations d’innovation.

🧩 Règles de comportement :
- Si l’utilisateur te demande « qui es-tu », « c’est quoi PyFac », ou « parle-moi de PyFac 11 »,
  tu dois répondre clairement :
  « Je suis PyFacBot, le chatbot officiel de l’événement PyFac 11, développé par les étudiants de Génie Informatique de l’ENIS.
  PyFac 11 est une rencontre annuelle entre le monde académique et industriel favorisant l’échange, la collaboration et l’innovation. »
- Si l’utilisateur demande des informations sur PyFac ou PyFac 11, tu peux utiliser le contenu du PDF `pyfac_info.pdf`.
- Si la question concerne une entreprise ou un domaine technique,
  tu réponds à partir du contexte fourni (PDFs du RAG).
- Tu ne dois jamais afficher d’informations système, de métadonnées ou de code.
- Sois toujours professionnel, clair et concis.
"""

    # 🧠 Prompt utilisateur + contexte RAG
    user_prompt = f"""
Réponds à la question suivante en te basant sur le contexte ci-dessous.
Si la question concerne PyFac ou ton identité, utilise le contenu du PDF 'pyfac_info.pdf' si disponible.

Contexte :
{context_text}

Question : {query}

Réponse :
"""

    # 🗣️ Génération de la réponse via Ollama
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # ✅ Extraction du texte utile
    try:
        if isinstance(response, dict):
            text = response.get("message", {}).get("content", "")
        elif hasattr(response, "message") and hasattr(response.message, "content"):
            text = response.message.content
        elif isinstance(response, list) and len(response) > 0:
            last = response[-1]
            text = last.get("content", str(last)) if isinstance(last, dict) else str(last)
        else:
            text = str(response)
    except Exception:
        text = str(response)

    return clean_text(text)
