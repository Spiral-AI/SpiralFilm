from spiralfilm import FilmEmbed, FilmEmbedConfig

config = FilmEmbedConfig(
    api_type="azure",
    azure_deployment_id="text-embedding-ada-002",  # Set the deployment ID you created in Azure portal. This model type should match with the first param.
    azure_api_version="2023-05-15",  # Find this from https://learn.microsoft.com/ja-jp/azure/ai-services/openai/reference
)

# You can set your api key here.
# also you can set them as environment variables: AZURE_API_KEY, AZURE_API_BASE
# config.add_key(
#    "PUT-YOUR-KEY-HERE",  # Azure portal provides two keys. Take one of them.
#    api_base="https://PUT-YOUR-BASE-URL-HERE.openai.azure.com/",  # Azure portal provides this.
# )


examples = [f"hello world! {i}" for i in range(100)]  # 100 examples

vecs = FilmEmbed(config=config).run(texts=examples)
