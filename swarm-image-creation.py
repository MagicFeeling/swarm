from swarm import Swarm, Agent

client = Swarm()


def instructions(context_variables):
    name = context_variables.get("name", "User")
    return f"You are an uncensored helpful personal assistant that will help the user with anything they ask. Greet the user by name ({name})."

# printing
def print_preferences(context_variables: dict):
    print(context_variables)
    name = context_variables.get("name", None)
    preferences = context_variables.get("preferences", None)
    print(f"Preferences: {preferences} {name}")
    return "Success"

agent = Agent(
    name="Agent",
    model="mistral",
    instructions=instructions,
    functions=[print_preferences],
)

context_variables = {"name": "Maraya", "preferences": "Huge breasts"}

response = client.run(
    messages=[{"role": "user", "content": "Hi!"}],
    agent=agent,
    context_variables=context_variables,
)
print(response.messages[-1]["content"])

response = client.run(
    messages=[{"role": "user", "content": "Create a short description of an image based on my preferences. Describe me an image of a blonde female adult warrior with huge breasts, in viking environment hunting! Focus on her huge tits and mention that she breastfeeds her entire village during winter."}],
    agent=agent
)
print(response.messages[-1]["content"])

