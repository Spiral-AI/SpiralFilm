from spiralfilm import FilmCore, FilmConfig

config = FilmConfig(max_queues=10)

placeholders_list = []
for i in range(20):
    placeholders_list.append({"number": str(i)})

f = FilmCore(
    prompt="""
Your lucky number is {{number}}.
""",
    config=config,
)

results = f.run_parallel(placeholders_list=placeholders_list)

print(results)
print(len(results))

print("----")
print(f.result_content)
