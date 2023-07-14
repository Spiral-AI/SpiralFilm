from spiralfilm import FilmCore, FilmConfig

# Ensure you have set the OPENAI_API_KEY environment variable
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

_template = """
Talk freely.
You're {{your_name}}, and I am {{my_name}}.
"""


f = FilmCore(prompt=_template)

# Show the list of placeholders in the template
print("placeholders =", f.placeholders())

variables = {"your_name": "Tom", "my_name": "John"}

# Show the number of tokens with the given variables
print("num_tokens =", f.num_tokens(variables), "max_tokens =", f.max_tokens())

# Generate a text with the given variables
result = f.run(variables)

# dump the result
f.summary(save_path="test.txt")
print(result)
