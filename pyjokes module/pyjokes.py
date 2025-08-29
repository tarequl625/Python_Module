# -------------------------------
# Random Programming Jokes Script
# -------------------------------

# Import the pyjokes library which contains a collection of programmer jokes
import pyjokes

# ------------------------------------
# Single Random Programming Joke
# ------------------------------------

# Get a single random programming joke
joke = pyjokes.get_joke(language='en', category='neutral')

# Print the single joke to the console
print("Single Joke:\n", joke)
# Example Output: "Why do programmers prefer dark mode? Because light attracts bugs."

# ------------------------------------
# Multiple Programming Jokes
# ------------------------------------

# Get a list of programming jokes (default is English and 'neutral' category)
jokes = pyjokes.get_jokes()

# Print a divider
print("\nMultiple Jokes:")

# Loop through the first 3 jokes and print them
for i in jokes[:3]:
    print("-", i)
    # Example Outputs:
    # - "A SQL query walks into a bar, goes up to two tables and says 'Can I join you?'"
    # - "How many programmers does it take to change a light bulb? None, that's a hardware problem."
    # - "There are only 10 kinds of people in this world: those who know binary and those who don't."