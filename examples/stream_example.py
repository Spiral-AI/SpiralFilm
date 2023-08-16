from spiralfilm import FilmCore
import sys

# Ensure you have set the OPENAI_API_KEY environment variable
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

f = FilmCore(
    prompt="""
Talk as you want.
You're {{user_name}}.
"""
)


for chunk in f.stream(placeholders={"user_name": "Tom"}):
    print(chunk, end="")
    sys.stdout.flush()
print()

f.summary(save_path="stream_summary.log")
