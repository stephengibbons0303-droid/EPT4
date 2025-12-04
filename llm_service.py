from openai import OpenAI

# UPDATED: Using the exact ID from your error log for Opus 4.5
DEFAULT_MODEL = "claude-opus-4-5"
DEFAULT_BASE_URL = "https://api.aimlapi.com/v1"

def call_llm(messages, api_key, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, max_tokens=4096):
    if not api_key:
        return "Error: API Key is missing."

    try:
        if not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"

        client = OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": messages[0]},
                {"role": "user", "content": messages[1]}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"
