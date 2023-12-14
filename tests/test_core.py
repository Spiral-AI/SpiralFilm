import pytest
from unittest.mock import patch, Mock
from spiralfilm.core import FilmCore
from spiralfilm.config import FilmConfig


# モックオブジェクトを使ってopenai APIの呼び出しをシミュレートするためのフィクスチャ
@pytest.fixture
def mock_openai_api():
    with patch("spiralfilm.core.openai") as mock:
        yield mock


# FilmCore クラスのインスタンスを生成するためのフィクスチャ
@pytest.fixture
def film_core_instance():
    prompt = "Hello, World! {{variable1}} is {{variable2}}."
    config = FilmConfig(model="gpt-4", temperature=0.9, max_tokens=100)
    return FilmCore(prompt, config=config)


# 環境変数を模擬するためのフィクスチャ
@pytest.fixture
def mock_env_vars():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-api-key"}):
        yield


# FilmCore.run メソッドのテスト
def test_run(film_core_instance, mock_openai_api, mock_env_vars):
    # 期待される応答をモックする
    expected_response = {
        "choices": [
            {
                "message": {"content": "Mocked response"},
                "finish_reason": "length",
                "index": 0,
            }
        ],
        "usage": {"completion_tokens": 17, "prompt_tokens": 57, "total_tokens": 74},
    }
    mock_openai_api.ChatCompletion.create.return_value = expected_response

    # placeholdersを用いてrunメソッドを呼び出す
    result = film_core_instance.run({"variable1": "summer", "variable2": "hot"})

    # 結果が期待通りであることをアサートする
    assert result == "Mocked response"
    assert film_core_instance.finished_reason == "length"
    assert film_core_instance.token_usages == expected_response["usage"]


# _placeholder メソッドのテスト
@pytest.mark.skip(reason="このテストは現在無効化されています")
def test_placeholder(film_core_instance):
    # プロンプトに含まれるプレースホルダーを置き換え
    original_prompt = film_core_instance.prompt
    placeholders = {"name": "Alice", "day": "Monday"}
    updated_prompt = film_core_instance._placeholder(original_prompt, placeholders)

    # 各プレースホルダーが適切に置き換えられたかをチェック
    for key, value in placeholders.items():
        assert value in updated_prompt


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
