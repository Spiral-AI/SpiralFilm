import pytest
import os
from unittest.mock import patch, MagicMock
from spiralfilm.core import FilmCore
from spiralfilm.config import FilmConfig


# 環境のセットアップ状況の確認
def test_env_setup():
    assert "OPENAI_API_KEY" in os.environ.keys()
    assert "OPENAI_API_KEY_1" in os.environ.keys()
    assert "OPENAI_API_KEY_2" in os.environ.keys()


# 最もシンプルな生成のテスト
def test_run():
    # 期待される応答をモックする: https://platform.openai.com/docs/api-reference/making-requests
    expected_response = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-1106",
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "choices": [
            {
                "message": {"role": "assistant", "content": "Yes"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    # FilmCore クラスのインスタンスを生成
    film_core_instance = FilmCore(
        prompt="""Hi {{name}}. Just answer 'Yes'""",
        config=FilmConfig(model="gpt-3.5-turbo-1106", temperature=0.0, max_tokens=100),
    )

    # placeholdersを用いてrunメソッドを呼び出す
    result = film_core_instance.run({"name": "Tom"})

    # 結果が期待通りであることをアサートする
    assert result == "Yes"
    assert film_core_instance.finished_reason == "stop"
    assert film_core_instance.token_usages == {
        "prompt_tokens": 25,
        "completion_tokens": 1,
        "total_tokens": 26,
    }


# _placeholder メソッドのテスト
def test_placeholder():
    # プロンプトに含まれるプレースホルダーを置き換え
    prompt = "{{variable1}} is {{variable2}}. And {{variable3}} is {{variable4}}."

    fc = FilmCore(prompt)

    # 正常系の確認
    placeholders = {
        "variable1": "summer",
        "variable2": "hot",
        "variable3": "winter",
        "variable4": "cold",
    }
    assert fc.placeholders() == ["variable1", "variable2", "variable3", "variable4"]
    assert fc._placeholder(prompt, placeholders) == "summer is hot. And winter is cold."

    # 異常系の確認 - variableが足りないケース
    placeholders = {"variable1": "summer", "variable2": "hot"}
    with pytest.raises(ValueError):
        fc._placeholder(prompt, placeholders)

    # 異常系の確認 - variableが間違っているケース
    placeholders = {
        "variable1": "summer",
        "variable2": "hot",
        "variable3": "winter",
        "hogehoge": "cold",
    }
    with pytest.raises(ValueError):
        fc._placeholder(prompt, placeholders)

    # 異常系の確認 - variableが余分なケース
    placeholders = {
        "variable1": "summer",
        "variable2": "hot",
        "variable3": "winter",
        "variable4": "cold",
        "hogehoge": "fugafuga",
    }
    with pytest.raises(ValueError):
        fc._placeholder(prompt, placeholders)


# _messages メソッドのテスト
@pytest.mark.skip(reason="このテストは現在無効化されています")
def test_messages(film_core_instance):
    # 正しいメッセージのリストが生成されるかをチェック
    history = ["User input", "Assistant response"]
    updated_messages = film_core_instance._messages(
        film_core_instance.prompt, history, film_core_instance.system_prompt
    )

    # messagesリストの構造と内容を検証
    assert len(updated_messages) == 3  # system_prompt + history + user_prompt
    assert updated_messages[0]["role"] == "system"
    assert updated_messages[1]["role"] == "user"
    assert updated_messages[2]["role"] == "assistant"


# その他のメソッドに対するテストも同様のパターンで追加することができます。
# 例えば num_tokens, max_tokens, summary などのメソッドです。

# テストの実行には、コマンドラインから `pytest` を実行します。
