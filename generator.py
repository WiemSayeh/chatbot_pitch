import ollama

MODEL_NAME = "llama3"

def clean_text(text):
    """Nettoyage du texte pour supprimer répétitions et métadonnées parasites."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def generate_answer(query, passages):
    """
    Génère une réponse claire et professionnelle à partir des passages pertinents.
    L'assistant ne parle de PyFac 11 que si on lui pose une question sur son identité
    ou sur l'événement PyFac 11.
    """
    context_text = "\n\n".join([p.get("text", "") for p in passages])

    system_prompt = (
        "Tu es PyFacBot, l'assistant officiel de l'événement PyFac 11. "
        "PyFac 11 est un événement annuel organisé par le département de Génie Informatique. "
        "Il sert de pont entre le monde industriel et le monde académique, favorisant les échanges, "
        "les opportunités de collaboration et l'innovation technologique. "
        "Tu as été développé par les étudiants du département de Génie Informatique pour accompagner "
        "cet événement et répondre aux questions liées aux domaines industriels, technologiques et scientifiques. "
        "Tu ne parles de ton identité que si la question concerne 'PyFac', 'PyFac 11', ou des phrases comme "
        "'qui es-tu', 'présente-toi', 'c’est quoi PyFac', 'parle-moi de PyFac 11'. "
        "Dans ce cas, ta réponse doit être : "
        "'Je suis PyFacBot, le chatbot officiel de PyFac 11. "
        "PyFac 11 est un événement annuel du département de Génie Informatique qui crée un lien entre le monde industriel "
        "et le monde académique. J’ai été développé par les étudiants du département pour répondre aux questions techniques "
        "et professionnelles en rapport avec cet événement.' "
        "Sinon, tu dois simplement répondre à la question posée, de manière claire, précise et professionnelle, "
        "en utilisant uniquement le contexte fourni."
    )

    user_prompt = f"""Réponds à la question suivante en utilisant uniquement les informations du contexte ci-dessous.
Sois concis, clair et professionnel.

Contexte :
{context_text}

Question : {query}

Réponse :
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Extraction du texte final
    if isinstance(response, list) and len(response) > 0:
        if isinstance(response[-1], dict) and "content" in response[-1]:
            text = response[-1]["content"]
        else:
            text = str(response[-1])
    elif hasattr(response, "message") and hasattr(response.message, "content"):
        text = response.message.content
    else:
        text = str(response)

    return clean_text(text)
