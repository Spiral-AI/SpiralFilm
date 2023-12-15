import pytest
from unittest.mock import patch, Mock
from spiralfilm.core import FilmCore
from spiralfilm.config import FilmConfig


# _placeholder メソッドのテスト
def test_placeholder(mock_openai_api):
    # プロンプトに含まれるプレースホルダーを置き換え
    prompt = "{{variable1}} is {{variable2}}. And {{variable3}} is {{variable4}}."
    placeholders = {
        "variable1": "summer",
        "variable2": "hot",
        "variable3": "winter",
        "variable4": "cold",
    }
    expected_prompt = "summer is hot. And winter is cold."

    fc = FilmCore(prompt)

    # 正常系の確認
    assert fc._placeholder(prompt, placeholders) == expected_prompt

    # 異常系の確認
    placeholders = {"variable1": "summer", "variable2": "hot"}
    # with pytest.raises(KeyError):
    fc._placeholder(prompt, placeholders)
