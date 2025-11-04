import ollama

MODEL_NAME = "llama3"

def clean_text(text):
    """Nettoyage du texte pour supprimer répétitions et métadonnées parasites."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def generate_answer(query, passages):
    """Génération de réponse propre à partir des passages pertinents."""
    context_text = "\n\n".join([p.get("text", "") for p in passages])
    prompt = f"""Réponds à la question suivante en utilisant uniquement les informations du contexte ci-dessous.
Sois concis, clair et lisible.

Contexte:
{context_text}

Question: {query}
Réponse:"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    # Extraire uniquement le texte de la réponse
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
