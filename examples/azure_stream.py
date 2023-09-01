from spiralfilm import FilmCore, FilmConfig
import sys

# Ensure you have set the OPENAI_API_KEY environment variable
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

config = FilmConfig(
    "gpt-3.5-turbo",  # Specify OpenAI-version name. This name will be used for various internal configurations such as max_tokens.
    api_type="azure",
    azure_deployment_id="gpt-35-turbo",  # Set the deployment ID you created in Azure portal. This model type should match with the first param.
    azure_api_version="2023-05-15",  # Find this from https://learn.microsoft.com/ja-jp/azure/ai-services/openai/reference
)

# You can set your api key here.
# also you can set them as environment variables: AZURE_API_KEY, AZURE_API_BASE
# config.add_key(
#    "PUT-YOUR-KEY-HERE",  # Azure portal provides two keys. Take one of them.
#    api_base="https://PUT-YOUR-BASE-URL-HERE.openai.azure.com/",  # Azure portal provides this.
# )


f = FilmCore(
    prompt="""
Talk as you want.
You're {{user_name}}.
"""
)


for chunk in f.stream(placeholders={"user_name": "Tom"}):
    print(chunk, end="")
    sys.stdout.flush()
print()

f.summary(save_path="stream_summary.log")
