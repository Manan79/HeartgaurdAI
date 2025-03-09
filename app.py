from openai import OpenAI
import streamlit as st

content = st.text_input("Enter your question here")

btn = st.button("Submit")

KEY = "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"

if btn:
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= KEY,
    )

    completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    },
    extra_body={},
    model="deepseek/deepseek-r1:free",
    messages=[
        {
        "role": "user",
        "content": content
        }
    ]
    )
    st.write(completion.choices[0].message.content)
    print(completion.choices[0].message.content)