from spiralfilm import FilmEmbed

# Ensure you have set the OPENAI_API_KEY environment variable
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

examples = []

examples.append("Today is a super good day.")
examples.append("Today is a good day.")
examples.append("Today is a bad day.")

vecs = FilmEmbed().run(texts=examples)


def calc_similarity(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])


print(
    f"Similarity between '{examples[0]}' and '{examples[1]}' : ",
    calc_similarity(vecs[0], vecs[1]),
)
print(
    f"Similarity between '{examples[0]}' and '{examples[2]}' : ",
    calc_similarity(vecs[0], vecs[2]),
)
