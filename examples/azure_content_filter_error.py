from spiralfilm import FilmCore, FilmConfig, ContentFilterError

# Basic concept of how to distribute the params:
# FilmConfig(...): Params which may impact the output quality.
# confi.add_key(...): Params which may be used in roundrobin mode.

config_azure = FilmConfig(
    "gpt-4",  # Specify OpenAI-version name. This name will be used for various internal configurations such as max_tokens.
    api_type="azure",
    azure_deployment_id="gpt-4",  # Set the deployment ID you created in Azure portal. This model type should match with the first param.
    azure_api_version="2023-05-15",  # Find this from https://learn.microsoft.com/ja-jp/azure/ai-services/openai/reference
)

config_openai = FilmConfig(
    "gpt-4",  # Specify OpenAI-version name. This name will be used for various internal configurations such as max_tokens.
    api_type="openai",
)

# You can set your api key here.
# also you can set them as environment variables: AZURE_API_KEY, AZURE_API_BASE
# config.add_key(
#    "PUT-YOUR-KEY-HERE",  # Azure portal provides two keys. Take one of them.
#    api_base="https://PUT-YOUR-BASE-URL-HERE.openai.azure.com/",  # Azure portal provides this.
# )

prompt = """
Convert the following input text to hiragana.

# Input text
109、知ってるよ。それは私のお気に入りのスポットなんだ

# Output text (Hiragana only)
"""

film_azure = FilmCore(
    prompt=prompt,
    config=config_azure,
)

film_openai = FilmCore(
    prompt=prompt,
    config=config_openai,
)


try:
    result = film_azure.run()
    print(result)
except ContentFilterError as err:
    result = film_openai.run()
    print(result)
