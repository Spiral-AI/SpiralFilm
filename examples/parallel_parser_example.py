from spiralfilm import FilmCore, FilmParser

_prompt = """
Make a simple {{format}} example. Topic is {{topic}}.
----
"""

format = "json"
placeholder_list = []
for topic in ["buidlings", "hobbies", "economics", "jobs"]:
    placeholder_list.append({"format": format, "topic": topic})

f = FilmCore(
    prompt=_prompt,
    parser=FilmParser(format=format),
).run_parallel(placeholders_list=placeholder_list)

print(f)
