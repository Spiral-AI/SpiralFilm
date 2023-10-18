from spiralfilm import TextCutter

text = """aaaaaaaaaaaaaaaaaaaaaaaaaaaaa
bbbbbbbbb
cccccccccccccccccccccccccccccc
dddd
eeeeeeeeeeeeeee
ffffffffff"""

print()
print("---- original text ----")
print(text)

print()
print("---- max_chars=20 ----")
print(TextCutter(text, max_chars=20))

print()
print("---- max_lines=3 ----")
print(TextCutter(text, max_lines=3))

print()
print("---- max_chars_in_line=10 ----")
print(TextCutter(text, max_chars_in_line=10))

print()
print("---- max_chars=20, max_lines=3 ----")
print(TextCutter(text, max_lines=3, max_chars_in_line=10))
print("----")
