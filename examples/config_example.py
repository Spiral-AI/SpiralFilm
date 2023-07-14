from spiralfilm import FilmCore, FilmConfig

# Ensure you have set the OPENAI_API_KEY environment variable
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

config = FilmConfig(model="gpt-4", temperature=0.5, max_tokens=100, use_cache=True)
_template = """
Talk as you want.
You're {{user_name}}.
"""

f = FilmCore(
    prompt=_template,
    config=config,
).run(placeholders={"user_name": "Tom"})

print(f)
