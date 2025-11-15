MODEL_HEAVY = "qwen2.5:latest"   # modèle RAG principal (précision)
MODEL_LIGHT = "phi3:latest"      # pour petites questions
EMBED_MODEL = "mxbai-embed-large"  # meilleur embedding open-source

MAX_CONTEXT_CHARS = 1600          # augmente pour meilleure qualité
MIN_SCORE = 0.15                  # plus strict pour éviter bruit
