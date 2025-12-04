from openai import OpenAI

def call_llm(messages, api_key, model="claude-4.5-opus", base_url="https://api.aimlapi.com/v1", max_tokens=4096):
    """
    Sends a message history to the LLM provider (AIMLAPI).
    """
    if not api_key:
        return "Error: API Key is missing. Please enter it in the sidebar."

    try:
        # Ensure URL ends in /v1 for OpenAI compatibility layer
        if not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Note: We do NOT use response_format={"type": "json_object"} here.
        # Anthropic models via proxy often error out with that flag.
        # We rely on the system prompt instructions ("OUTPUT FORMAT: JSON") which Opus handles perfectly.
        
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
