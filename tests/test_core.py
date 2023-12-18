import pytest
import os
from unittest.mock import patch, MagicMock
from spiralfilm.core import FilmCore
from spiralfilm.config import FilmConfig
import asyncio


# 環境のセットアップ状況の確認
# def test_env_setup():
#    assert "OPENAI_API_KEY" in os.environ.keys()
#    assert "OPENAI_API_KEY_1" in os.environ.keys()
#    assert "OPENAI_API_KEY_2" in os.environ.keys()
#    assert "AZUREOPENAI_API_KEY" in os.environ.keys()


# 最もシンプルな生成のテスト
@pytest.mark.parametrize("model", ["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview"])
@pytest.mark.parametrize("temperature", [0.0, 0.00001])
@pytest.mark.parametrize("mode", ["run", "stream", "stream_async", "run_async"])
def test_run(model, temperature, mode):
    # FilmCore クラスのインスタンスを生成
    film_core_instance = FilmCore(
        prompt="""Hi {{name}}. Just answer 'Yes.'""",
        config=FilmConfig(model=model, temperature=temperature, max_tokens=3),
    )

    # placeholdersを用いてrunメソッドを呼び出す
    if mode == "run":
        result = film_core_instance.run({"name": "Tom"})
    elif mode == "run_async":
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(film_core_instance.run_async({"name": "Tom"}))
    elif mode == "stream":
        result = ""
        for t in film_core_instance.stream({"name": "Tom"}):
            result += t
    elif mode == "stream_async":
        loop = asyncio.get_event_loop()
        result = ""
        async_gen = film_core_instance.stream_async({"name": "Tom"})
        while True:
            try:
                result += loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break

    # 結果が期待通りであることをアサートする
    assert result == "Yes."
    assert film_core_instance.finish_reason == "stop"
    assert film_core_instance.token_usages["prompt_tokens"] > 0
    assert film_core_instance.token_usages["completion_tokens"] > 0
    assert film_core_instance.token_usages["total_tokens"] > 0


# ラウンドロビンをテスト
def test_roundrobin():
    config = FilmConfig()
    config.add_key(os.environ["OPENAI_API_KEY_1"])
    config.add_key(os.environ["OPENAI_API_KEY_2"])
    # FilmCore インスタンスの run メソッドをモック化
    with patch.object(FilmCore, "run", return_value=None) as mock_run:
        f = FilmCore(
            prompt="""
    Talk as you want.
    You're {{user_name}}.
    """,
            config=config,
        )

        # run メソッドを10回呼び出し
        for _ in range(10):
            f.run(placeholders={"user_name": "Tom"})

        # APIキーがラウンドロビン方式で使用されていることを確認するために、config.apikeysのリストを、"available_time"でソートし、一番最初の要素のapi_keyを取得
        api_keys_used = [
            api_key["api_key"]
            for api_key in sorted(f.config.apikeys, key=lambda x: x["available_time"])
        ]
        # 交互に登場しているか確認
        for i in range(0, len(api_keys_used), 2):
            assert api_keys_used[i] != api_keys_used[i + 1]

        # run メソッドが10回呼ばれたことを確認
        assert mock_run.call_count == 10


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
