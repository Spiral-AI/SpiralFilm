from spiralfilm import FilmCore

# Ensure you have set the OPENAI_API_KEY environment variable
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

fc1 = FilmCore(
    prompt="""
Remember that x={{num}}.
Hello!
"""
)
print(fc1.run(placeholders={"num": "1234"}))

fc2 = FilmCore.create_from(
    fc1,
    prompt="""
Do you remember x?
                         """,
)

print(fc2.run())
