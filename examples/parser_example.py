from spiralfilm import FilmCore, FilmParser

_prompt = """
Make a simple json example.
----
"""

# This generates addional texts so difficult to parse.
# f = FilmCore(
#    prompt=_prompt,
# ).run()
# print(f)

# But this automatically fix the format.
f = FilmCore(
    prompt=_prompt,
    parser=FilmParser(format="json"),
).run()

print(f)
