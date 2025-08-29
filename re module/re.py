import re

# Sample text
txt = """Amount [400$]\n
                of 96!"""

# 1. Find all occurrences of 'n'
a = re.findall("n", txt)
print(a)   # ['n', 'n']

# 2. Match all non-whitespace characters
b = re.findall("\S", txt)
print(b)   # ['A','m','o','u','n','t','[','4','0','0','$',']','o','f','9','6','!']

# 3. Match all whitespace characters
c = re.findall("\s", txt)
print(c)   # [' ', '\n']

# 4. Match all digits
d = re.findall("\d", txt)
print(d)   # ['4','0','0','9','6']

# 5. Match all non-digits
e = re.findall("\D", txt)
print(e)   # ['A','m','o','u','n','t',' ', '[', '$', ']', '\n','o','f',' ','!']

# 6. Match all word characters (letters, digits, underscore)
f = re.findall("\w", txt)
print(f)   # ['A','m','o','u','n','t','4','0','0','o','f','9','6']

# 7. Match all non-word characters
g = re.findall("\W", txt)
print(g)   # [' ', '[', '$', ']', '\n', ' ', '!']

# 8. Match '96!' only at end of string
h = re.findall("96!\\Z", txt)
print(h)   # ['96!']

# 9. Match specific letters 'a', 'o', 'f'
s = re.findall("[aof]", txt)
print(s)   # ['A','o','o','f']

# 10. Match all except 'a', 'o', 'f'
t = re.findall("[^aof]", txt)
print(t)   # ['m','u','n','t',' ','[','4','0','0','$',']','\n','9','6','!']

# 11. Match lowercase letters a-m
i = re.findall("[a-m]", txt)
print(i)   # ['m','a','n','o','f']

# 12. Match all letters
u = re.findall("[a-zA-Z]", txt)
print(u)   # ['A','m','o','u','n','t','o','f']

# 13. Match 'A' followed by any 4 chars then 't'
j = re.findall("A....t", txt)
print(j)   # ['Amount']

# 14. Match 'A' at start of string
k = re.findall("^A", txt)
print(k)   # ['A']

# 15. Match '!' at end of string
l = re.findall("!$", txt)
print(l)   # ['!']

# 16. Match 'A' followed by any chars until 't'
m = re.findall("A.*t", txt)
print(m)   # ['Amount']

# 17. Match 'A' at start of string (alternative)
n = re.findall("\\AA", txt)
print(n)   # ['A']

# 18. Match 'f' at end of a word
o = re.findall(r"f\\b", txt)
print(o)   # ['f']

# 19. Match 'A' followed by exactly 4 chars then 't'
p = re.findall("A.{4}t", txt)
print(p)   # ['Amount']

# 20. Match either 'of' or 'in'
q = re.findall("of|in", txt)
print(q)   # ['of']

# 21. Match 'of' at start of line
r = re.findall("^of", txt, re.M)
print(r)   # ['of']

# 22. Search for first occurrence of 'in'
txt2 = "The rain in Spain"
x = re.search("in", txt2)
print(x.group())   # 'in'
print(x.span())    # (5, 7)
print(x.start())   # 5
print(x.end())     # 7
print(x.string)    # 'The rain in Spain'

# 23. Split string at whitespaces
x = re.split("\s", txt2)
print(x)   # ['The', 'rain', 'in', 'Spain']

# 24. Replace whitespaces with '9'
x = re.sub("\s", "9", txt2)
print(x)   # 'The9rain9in9Spain'