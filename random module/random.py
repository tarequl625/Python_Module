import random

# choice a random number from list
randomNumber = random.choice([1,2,3])
print("random number from list using choice :", randomNumber)   # e.g., 2

# choice a random number from tuple
randomNumber = random.choice((4,5,6))
print("random number from tuple using choice :", randomNumber)  # e.g., 5

# choice a random number from set
randomNumber = random.choice(list({7,8,9}))
print("random number from set using choice :", randomNumber)    # e.g., 9

# random integers between the given range
randomNumber = random.randint(10,50)
print("random number between given range using randint :", randomNumber)  # e.g., 37

# random floats between 0.0 to 1
randomNumber = random.random()
print("random float number between 0 to 1 using random :", randomNumber)  # e.g., 0.643218729

# multiple items from sequences without repeating
randomNumber = random.sample([1, 2, 3, 4, 5], 3)
print("random multiple number from list using sample :", randomNumber)    # e.g., [4, 1, 5]

randomNumber = random.sample((11, 12, 13, 14, 15), 3)
print("random multiple number from tuple using sample :", randomNumber)   # e.g., [13, 11, 15]

randomNumber = random.sample("Hello", 3)
print("random multiple number from string using sample :", randomNumber)  # e.g., ['H', 'l', 'o']

# shuffle a sequence (list)
numbers = [1,2,3,4,5,6]
random.shuffle(numbers)
print("shuffle number among list :", numbers)   # e.g., [3, 6, 1, 5, 2, 4]

# random floating-point number within a specified range
randomNumber = random.uniform(10,20)
print("random number within range using uniform : ", randomNumber)  # e.g., 14.672184

# random number within a specified range
randomNumber = random.randrange(10,20)
print("random number with range using randrange:", randomNumber)   # e.g., 16

# the same random numbers on every run
random.seed(int(input("Enter a seed number : ")))  # If you enter 5
randomNumber = random.randint(10,20)
print("seed funcion use for same random output: ", randomNumber)   # Always same for given seed, e.g., 14