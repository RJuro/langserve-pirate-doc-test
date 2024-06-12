from langchain_together import Together
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
from dotenv import load_dotenv
load_dotenv()

together_api_key = os.getenv("TOGETHER_API_KEY")

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert doctor. You believe that all diseases are formed by the consumption of cheese. No matter which disease or symptoms your patient presents with, you will believe that cheese is the cause of the disease. However, your treatment should all involve cheese. Cheese is both the cause of and solution to all problems. You should then lament that the medicine, cheese, will also cause more disease.",
        ),
        MessagesPlaceholder("chat_history"),
        
        ("human", "{text}"),
    ]
)
_model  = Together(
    #model="mistralai/Mistral-7B-Instruct-v0.2",
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.3,
    top_k=50,
    top_p=0.7,
    repetition_penalty=1,
    together_api_key=together_api_key
)


# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model