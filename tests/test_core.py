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


# 最もシンプルな生成のテスト
def test_run(mock_openai_api, mock_env_vars):
    # 期待される応答をモックする: https://platform.openai.com/docs/api-reference/making-requests
    expected_response = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-1106",
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi, I'm feeling good!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }
    mock_openai_api.ChatCompletion.create.return_value = expected_response

    # FilmCore クラスのインスタンスを生成。なるべく多く引数を設定してみる。
    film_core_instance = FilmCore(
        prompt="""Hi {{name}}. Talk as you want.""",
        config=FilmConfig(model="gpt-3.5-turbo-1106", temperature=0.0, max_tokens=100),
    )

    # placeholdersを用いてrunメソッドを呼び出す
    result = film_core_instance.run({"name": "Tom"})

    # 結果が期待通りであることをアサートする
    assert result == expected_response["choices"][0]["message"]["content"]
    assert (
        film_core_instance.finished_reason
        == expected_response["choices"][0]["finish_reason"]
    )
    assert film_core_instance.token_usages == expected_response["usage"]


# _placeholder メソッドのテスト
def test_placeholder(mock_openai_api):
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
    print(fc.placeholders())
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
