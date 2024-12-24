import unittest
from spiralfilm.core import FilmCore

class TestIntegration(unittest.TestCase):
    def test_openai_api(self):
        core = FilmCore(prompt="Hello!")
        response = core.run({"name": "Test"})
        self.assertIn("Hello", response)

def test_simple_chat():
    core = FilmCore(prompt="What is AI?")
    response = core.run(placeholders={})
    assert "artificial intelligence" in response.lower()

def test_placeholder_chat():
    core = FilmCore(prompt="What is {{topic}}?")
    response = core.run(placeholders={"topic": "AI"})
    assert "AI" in response

def test_conversation_context():
    core = FilmCore(prompt="Who won the World Series in 2020?")
    core.run(placeholders={})
    new_core = FilmCore.create_from(core, prompt="Where was it played?")
    response = new_core.run(placeholders={})
    assert "Dodgers" in response or "Los Angeles" in response
