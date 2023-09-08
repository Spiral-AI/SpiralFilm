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

f = FilmCore(
    prompt=_prompt,
).run()

print(f)

# But this automatically fix the format.
f = FilmCore(
    prompt=_prompt,
    parser=FilmParser(format="json"),
).run()

print(f)


_prompt = """
Come up with 5 name candidates of my kid.
Start each line with "- ".
Add a comment after listing up all the names.
----
"""

f = FilmCore(
    prompt=_prompt,
).run()

print(f)

f = FilmCore(
    prompt=_prompt,
    parser=FilmParser(format="lines", line_prefix="-"),
).run()

print(f)
