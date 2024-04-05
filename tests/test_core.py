import pytest
import os
import sys

sys.path.append(".")
from unittest.mock import patch
from spiralfilm.core import FilmCore
from spiralfilm.config import FilmConfig
from spiralfilm.embed import FilmEmbed, FilmEmbedConfig
from openai import OpenAI
import nest_asyncio

nest_asyncio.apply()

assert os.environ["OPENAI_API_KEY"] is not None
assert os.environ["OPENAI_API_KEY_1"] is not None
assert os.environ["OPENAI_API_KEY_2"] is not None
assert os.environ["AZURE_API_KEY"] is not None
assert os.environ["AZURE_API_BASE"] is not None


# 最もシンプルな生成のテスト
# @pytest.mark.skip(reason="このテストは現在スキップされています")
@pytest.mark.parametrize("model", ["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview"])
@pytest.mark.parametrize("temperature", [0.0, 0.00001])
@pytest.mark.parametrize("mode", ["stream_async", "run_async"])
@pytest.mark.asyncio
async def test_run_async(model, temperature, mode):
    # FilmCore クラスのインスタンスを生成
    film_core_instance = FilmCore(
        prompt="""Hi {{name}}. Just answer 'Yes.'""",
        config=FilmConfig(model=model, temperature=temperature, max_tokens=3),
    )

    # placeholdersを用いてrunメソッドを呼び出す
    if mode == "run_async":
        result = await film_core_instance.run_async({"name": "Tom"})
    elif mode == "stream_async":
        result = ""
        async_gen = film_core_instance.stream_async({"name": "Tom"})
        async for t in async_gen:
            result += t

    # 結果が期待通りであることをアサートする
    assert result == "Yes."
    assert film_core_instance.finish_reason == "stop"
    assert film_core_instance.token_usages["prompt_tokens"] > 0
    assert film_core_instance.token_usages["completion_tokens"] > 0
    assert film_core_instance.token_usages["total_tokens"] > 0


# @pytest.mark.skip(reason="このテストは現在スキップされています")
@pytest.mark.parametrize("model", ["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview"])
@pytest.mark.parametrize("temperature", [0.0, 0.00001])
@pytest.mark.parametrize("mode", ["run", "stream"])
async def test_run(model, temperature, mode):
    # FilmCore クラスのインスタンスを生成
    film_core_instance = FilmCore(
        prompt="""Hi {{name}}. Just answer 'Yes.'""",
        config=FilmConfig(model=model, temperature=temperature, max_tokens=3),
    )

    # placeholdersを用いてrunメソッドを呼び出す
    if mode == "run":
        result = film_core_instance.run({"name": "Tom"})
    elif mode == "stream":
        result = ""
        for t in film_core_instance.stream({"name": "Tom"}):
            result += t

    # 結果が期待通りであることをアサートする
    assert result == "Yes."
    assert film_core_instance.finish_reason == "stop"
    assert film_core_instance.token_usages["prompt_tokens"] > 0
    assert film_core_instance.token_usages["completion_tokens"] > 0
    assert film_core_instance.token_usages["total_tokens"] > 0


# Azureのテスト
# @pytest.mark.skip(reason="このテストは現在スキップされています")
@pytest.mark.asyncio
async def test_azure_run_async():
    # FilmCore クラスのインスタンスを生成
    film_core_instance = FilmCore(
        prompt="""Hi {{name}}. Just answer 'Yes.'""",
        config=FilmConfig(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=3,
            api_type="azure",
            azure_deployment_id="gpt-35-turbo",
            azure_api_version="2023-05-15",
        ),
    )
    film_core_instance.config.add_key(
        os.environ["AZURE_API_KEY"], api_base=os.environ["AZURE_API_BASE"]
    )
    result = await film_core_instance.run_async({"name": "Tom"})
    # 結果が期待通りであることをアサートする
    assert result == "Yes."
    assert film_core_instance.finish_reason == "stop"
    assert film_core_instance.token_usages["prompt_tokens"] > 0
    assert film_core_instance.token_usages["completion_tokens"] > 0
    assert film_core_instance.token_usages["total_tokens"] > 0


# @pytest.mark.skip(reason="このテストは現在スキップされています")
def test_azure_run():
    # FilmCore クラスのインスタンスを生成
    film_core_instance = FilmCore(
        prompt="""Hi {{name}}. Just answer 'Yes.'""",
        config=FilmConfig(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=3,
            api_type="azure",
            azure_deployment_id="gpt-35-turbo",
            azure_api_version="2023-05-15",
        ),
    )
    film_core_instance.config.add_key(
        os.environ["AZURE_API_KEY"], api_base=os.environ["AZURE_API_BASE"]
    )
    result = film_core_instance.run({"name": "Tom"})
    assert result == "Yes."
    assert film_core_instance.finish_reason == "stop"
    assert film_core_instance.token_usages["prompt_tokens"] > 0
    assert film_core_instance.token_usages["completion_tokens"] > 0
    assert film_core_instance.token_usages["total_tokens"] > 0


# 最もシンプルな生成のテスト
def test_cache():
    # FilmCore クラスのインスタンスを生成
    f = FilmCore(
        prompt="""Hi {{name}}. Just answer 'Yes.'""",
        config=FilmConfig(
            model="gpt-3.5-turbo",
            use_cache=True,
        ),
    )
    result1 = f.run({"name": "Tom"})
    result2 = f.run({"name": "Tom"})

    # 結果が期待通りであることをアサートする
    assert result1 == "Yes."
    assert result2 == "Yes."


def test_embed():
    examples = ["Today is a super good day." for _ in range(3)]

    vecs = FilmEmbed().run(texts=examples)

    # 結果が期待通りであることをアサートする
    assert len(vecs) == len(examples)


@pytest.mark.asyncio
async def test_azure_embed_run_async():
    examples = ["Today is a super good day." for _ in range(3)]

    config = FilmEmbedConfig(
        api_type="azure",
        azure_deployment_id="text-embedding-ada-002",  # Set the deployment ID you created in Azure portal. This model type should match with the first param.
        azure_api_version="2023-05-15",  # Find this from https://learn.microsoft.com/ja-jp/azure/ai-services/openai/reference
    )

    config.add_key(os.environ["AZURE_API_KEY"], api_base=os.environ["AZURE_API_BASE"])

    vecs = await FilmEmbed(config=config).run_async(texts=examples)

    # 結果が期待通りであることをアサートする
    assert len(vecs) == len(examples)


def test_azure_embed_run():
    examples = ["Today is a super good day." for _ in range(3)]

    config = FilmEmbedConfig(
        api_type="azure",
        azure_deployment_id="text-embedding-ada-002",  # Set the deployment ID you created in Azure portal. This model type should match with the first param.
        azure_api_version="2023-05-15",  # Find this from https://learn.microsoft.com/ja-jp/azure/ai-services/openai/reference
    )

    config.add_key(os.environ["AZURE_API_KEY"], api_base=os.environ["AZURE_API_BASE"])

    vecs = FilmEmbed(config=config).run(texts=examples)

    # 結果が期待通りであることをアサートする
    assert len(vecs) == len(examples)


def test_embed_cache():
    examples = [f"Hello! {i}" for i in range(5)]
    # t0 = time.time()
    vecs1 = FilmEmbed(
        config=FilmEmbedConfig(
            use_cache=True,
        )
    ).run(texts=examples)
    # t1 = time.time()
    vecs2 = FilmEmbed(
        config=FilmEmbedConfig(
            use_cache=True,
        )
    ).run(texts=examples)
    # t2 = time.time()

    # 結果が期待通りであることをアサートする
    assert len(vecs1) == len(examples)
    assert len(vecs2) == len(examples)


# tokenカウントをテスト
# @pytest.mark.skip(reason="このテストは現在スキップされています")
@pytest.mark.parametrize(
    "model",
    [
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo",
        # "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4",
    ],
)
def test_token_count(model):
    messages = []
    system_prompt = "あなたはSmart Assistantです。名前は「アイ」です。"
    messages.append(
        {"role": "assistant", "content": "私はアイと言います。楽しく会話しましょう。"}
    )
    messages.append(
        {
            "role": "user",
            "content": "分かりました。私は田中です。あなたのフルネームを教えてください。",
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": "私の名前はアイです。私はあなたのAIアシスタントです。",
        }
    )
    dummy_prompt = "Just answer 'I'm AI. Nice to meet you.'"
    # リファレンス用の文字列生成
    chat_completion = OpenAI().chat.completions.create(
        messages=messages
        + [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": dummy_prompt},
        ],
        model=model,
        temperature=0.0,
        max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
    )

    prompt_tokens = chat_completion.usage.prompt_tokens
    total_tokens = chat_completion.usage.total_tokens

    # FilmCore クラスのインスタンスを生成
    film_core_instance = FilmCore(
        prompt=dummy_prompt,
        history=messages,
        system_prompt=system_prompt,
        config=FilmConfig(model=model, temperature=0.0, max_tokens=1),
    )

    # placeholdersを用いてrunメソッドを呼び出す
    film_core_instance.run()

    # 結果が期待通りであることをアサートする
    def compare(v1, v2, threshold=0.1):
        return abs(v1 - v2) / max(v1, v2) <= threshold

    assert compare(
        film_core_instance.token_usages["prompt_tokens"], prompt_tokens, threshold=0.05
    )
    assert compare(
        film_core_instance.token_usages["total_tokens"], total_tokens, threshold=0.05
    )


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

        # run メソッドを3回呼び出し
        for _ in range(3):
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
        assert mock_run.call_count == 3


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
