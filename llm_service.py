from openai import OpenAI

# Standard OpenAI Default
DEFAULT_MODEL = "gpt-4o" 

def call_llm(messages, api_key, model=DEFAULT_MODEL, base_url=None, max_tokens=4096):
    """
    Sends a message history to the OpenAI API.
    """
    if not api_key:
        return "Error: API Key is missing. Please enter it in the sidebar."

    try:
        # Standard OpenAI Client (no base_url needed unless using a proxy)
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": messages[0]},
                {"role": "user", "content": messages[1]}
            ],
            temperature=0.7,
            max_tokens=max_tokens,
            # OpenAI specific flag to force valid JSON
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"
