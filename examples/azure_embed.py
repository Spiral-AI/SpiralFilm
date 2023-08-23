from spiralfilm import FilmEmbed, FilmEmbedConfig

# Basic concept of how to distribute the params:
# FilmConfig(...): Params which may impact the output quality.
# confi.add_key(...): Params which may be used in roundrobin mode.

config = FilmEmbedConfig(
    api_type="azure",
    azure_deployment_id="text-embedding-ada-002",  # Set the deployment ID you created in Azure portal. This model type should match with the first param.
    azure_api_version="2023-05-15",  # Find this from https://learn.microsoft.com/ja-jp/azure/ai-services/openai/reference
)

config.add_key(
    "PUT-YOUR-KEY-HERE",  # Azure portal provides two keys. Take one of them.
    api_base="https://PUT-YOUR-BASE-URL-HERE.openai.azure.com/",  # Azure portal provides this.
)


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
