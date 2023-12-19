import pytest
from unittest.mock import patch, Mock
from spiralfilm.core import FilmCore
from spiralfilm.config import FilmConfig
from openai import OpenAI

chat_completion = OpenAI().chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "あなたは賢いAIアシスタントです。名前は、「アイ」です。",
        },
        {
            "role": "assistant",
            "content": "楽しく会話しましょう。",
        },
        {
            "role": "user",
            "content": """分かりました。あなたの名前を教えてください。出力は、以下のフォーマットでお願いします。
            「こんにちは。私の名前は(AIの名前)と言います。皆さんにお会いできて嬉しいです。これから、様々なことを教えて下さいね。」
            """,
        },
    ],
    model="gpt-4",
)
print(chat_completion.usage)
