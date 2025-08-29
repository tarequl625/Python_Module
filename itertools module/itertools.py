import itertools

# -------------------------------
# itertools.count(start=0, step=1)
# Infinite counter generator
# -------------------------------
for i in itertools.count(10, 2):
    if i > 50:
        break
    print(i)  # Output: 10, 12, 14, ..., 50

# -------------------------------
# itertools.cycle(iterable)
# Infinite loop through iterable
# -------------------------------
values = [40, 50, 60]
count = 0
for i in itertools.cycle(values):
    print(i)
    count += 1
    if count > 1:
        break
# Output: 40, 50

# -------------------------------
# itertools.repeat(object, times)
# -------------------------------
for i in itertools.repeat("Hello", 3):
    print(i)
# Output: Hello, Hello, Hello

# -------------------------------
# itertools.permutations(iterable, r)
# -------------------------------
for i in itertools.permutations(["A", "B", "C"], 2):
    print(i)
# Output: ('A','B'), ('A','C'), ('B','A'), ...

# -------------------------------
# itertools.combinations(iterable, r)
# -------------------------------
for i in itertools.combinations(["A", "B", "C"], 2):
    print(i)
# Output: ('A','B'), ('A','C'), ('B','C')

# -------------------------------
# itertools.combinations_with_replacement(iterable, r)
# -------------------------------
for i in itertools.combinations_with_replacement(["A", "B", "C"], 2):
    print(i)
# Output: ('A','A'), ('A','B'), ('A','C'), ...

# -------------------------------
# itertools.product(iterables, repeat=1)
# -------------------------------
for i in itertools.product(["A","B","C"], [1,2,3]):
    print(i)
# Output: ('A',1), ('A',2), ..., ('C',3)

# -------------------------------
# itertools.chain(*iterables)
# -------------------------------
for i in itertools.chain(["A","B","C"], [1,2,3]):
    print(i)
# Output: A, B, C, 1, 2, 3

# -------------------------------
# itertools.compress(data, selectors)
# -------------------------------
for i in itertools.compress(["A","B","C"], [True, False, True]):
    print(i)
# Output: A, C

# -------------------------------
# itertools.accumulate(iterable)
# -------------------------------
for i in itertools.accumulate([2,3,5,7]):
    print(i)
# Output: 2, 5, 10, 17

# -------------------------------
# itertools.islice(iterable, start, stop, step)
# -------------------------------
for i in itertools.islice([2,3,4,5,6,7,8,9], 1, 9, 1):
    print(i)
# Output: 3,4,5,6,7,8,9

# -------------------------------
# itertools.groupby(iterable, key)
# -------------------------------
data = [2, 4, 6, 7, 8, 10, 11, 12, 13]
g_data = itertools.groupby(data, key=lambda x: 'Even' if x%2==0 else 'Odd')
for key, group in g_data:
    print(f"Group: {key} -> {list(group)}")
# Output: Group: Even -> [2,4,6], Group: Odd -> [7], ...

# -------------------------------
# itertools.filterfalse(func, iterable)
# -------------------------------
print(list(itertools.filterfalse(lambda x: x>10, [2,5,8,22,55,88])))
# Output: [2,5,8]

# -------------------------------
# itertools.batched(iterable, n)
# -------------------------------
print(list(itertools.batched("Hello",2)))
# Output: [('H','e'), ('l','l'), ('o',)]

# -------------------------------
# itertools.pairwise(iterable)
# -------------------------------
print(list(itertools.pairwise("Hello")))
# Output: [('H','e'), ('e','l'), ('l','l'), ('l','o')]

# -------------------------------
# itertools.zip_longest(*iterables, fillvalue)
# -------------------------------
print(list(itertools.zip_longest("Hello", [2,3], fillvalue="-")))
# Output: [('H',2), ('e',3), ('l','-'), ('l','-'), ('o','-')]

# -------------------------------
# itertools.dropwhile(func, iterable)
# -------------------------------
print(list(itertools.dropwhile(lambda x: x%5==0, [5,10,15,20,25,30,7,8])))
# Output: [7,8]  (drops all divisible by 5 at the start)

# -------------------------------
# itertools.takewhile(func, iterable)
# -------------------------------
print(list(itertools.takewhile(lambda x: x%5==0, [5,10,15,20,25,30,7,8])))
# Output: [5,10,15,20,25,30] (takes elements until predicate False)

# -------------------------------
# itertools.tee(iterable, n)
# -------------------------------
li = [2,4,6,7,8,10,20]
tee = itertools.tee(li, 2)
for i in range(2):
    print(list(tee[i]))
# Output: [2,4,6,7,8,10,20] repeated twice