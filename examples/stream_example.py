from spiralfilm import FilmCore

# Ensure you have set the OPENAI_API_KEY environment variable
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

f = FilmCore(
    prompt="""
Talk as you want.
You're {{user_name}}.
"""
).stream(placeholders={"user_name": "Tom"})

for chunk in f:
    print(chunk, end="")
print()
