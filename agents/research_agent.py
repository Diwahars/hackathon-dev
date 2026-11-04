from bifrost_client import chat_text

def run(query):
    return chat_text([
        {"role": "system", "content":
         "You are a careful research assistant. Answer clearly. Do not invent citations. "
         "For latest/current questions, acknowledge that live evidence may be needed."},
        {"role": "user", "content": query},
    ])
