from spiral_film import FilmCore

# Ensure you have set the OPENAI_API_KEY environment variable
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

f = FilmCore(
    prompt="""
Talk as you want.
You're {{user_name}}.
"""
).run(placeholders={"user_name": "Tom"})

print(f)
