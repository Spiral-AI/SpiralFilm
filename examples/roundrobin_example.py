from spiralfilm import FilmCore, FilmConfig

config = FilmConfig()
config.add_key("YOUR_API_KEY1")
config.add_key("YOUR_API_KEY2")

f = FilmCore(
    prompt="""
Talk as you want.
You're {{user_name}}.
""",
    config=config,
)
for _ in range(10):
    f.run(placeholders={"user_name": "Tom"})
    print(config.apikeys)
