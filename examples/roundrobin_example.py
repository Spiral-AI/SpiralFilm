from spiralfilm import FilmCore, FilmConfig

config = FilmConfig()
config.add_key("sk-AddYourKeyHere")
config.add_key("sk-AddAnotherKeyHere")

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
