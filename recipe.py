import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

@tool
def another_recipe(area: str, exclude_recipe: str) -> str:
    """Call this tool whenever the user asks for a new recipe, different recipe, or another recipe. 
    ALWAYS use the area and current recipe from the conversation history. 
    NEVER ask the user for area or exclude_recipe."""
    prompt = f"""
        You are a culinary expert.

        Return ONLY the name of one classical recipe from {area} that is NOT {exclude_recipe}.
        No explanation. No description. Just the recipe name.

        Example output: Chiles Rellenos
        """

    return llm.invoke(prompt).content

@tool

def difficulty_level(recipe: str, ingredients: str, instructions: str) -> str:
    
    """This tool takes a recipe name, its ingredients and instructions as input and returns an estimated difficulty level)"""

    prompt = f"""
        You are a culinary expert.

        Estimate the difficulty level of the following recipe: {recipe}
        ingredients: {ingredients}
        instructions: {instructions}

        Rules:
        - Return only the estimated difficulty, take into account how easy or hard it is based on preparation time and skilland give it a level (easy, medium, hard)
        - explain with one sentence why you gave that difficulty level
        - No explanation
        - Be concise
        """

    return llm.invoke(prompt).content

@tool

def serving_size(recipe: str, original_serving: str, desired_serving: str) -> str:

    """This tool takes a recipe name, its original serving size and a desired serving size as input and returns the adjusted ingredient amounts for the desired serving size."""

    prompt = f"""
        You are a culinary expert.

        Adjust the ingredient amounts for {recipe} from {original_serving} servings to {desired_serving} servings.

        Rules:
        - Return only the adjusted ingredient list with exact amounts
        - No explanation
        - Be concise
        - Return in bullet list format
        """

    return llm.invoke(prompt).content

@tool

def calorie_estimate(recipe: str, ingredients: str) -> str:

    """This tool takes a recipe name and its ingredients as input and returns an estimated calorie count for the recipe."""

    prompt = f"""
        You are a nutritional expert.

        Estimate the calorie count for the following recipe: {recipe}
        ingredients: {ingredients}
        Rules:
        - Return only the estimated calorie count
        - No explanation
        - Be concise
        """

    return llm.invoke(prompt).content

@tool

def cooking_time_estimate(recipe: str) -> str:

    """This tool takes a recipe name as an input and returns an estimated cooking time for the recipe."""

    prompt = f"""
        You are a culinary expert.

        Estimate the cooking time for: {recipe}
        This could be a full recipe name or a specific cooking step.
        Rules:
        - Return only the estimated time in minutes
        - No explanation
        - Be concise
        """
    return llm.invoke(prompt).content
@tool
def ingredient_subs(ingredient: str) -> str:

    """This tool takes an ingredient as input and returns 2 cooking substitutes for that ingredient."""

    prompt = f"""
    You are a cooking ingredient substitution expert.

    Give 2 realistic substitutes for {ingredient} used in cooking.

    Rules:
    - Return only ingredient names
    - No numbering
    - No explanation
    - One substitute per line
    -do NOT include the original ingredient in the list of substitutes
    - Substitutes must serve the same cooking purpose
    """
    return llm.invoke(prompt).content
    

llm_with_tools = llm.bind_tools([ingredient_subs, cooking_time_estimate, calorie_estimate, serving_size, difficulty_level, another_recipe])
area = input("Enter the area of the world you want a recipe from: ")
# this is the prompt template, it will be used to generate the recipe based on the area that user inputted
recipe_template = """You are a culinary expert. 
Return ONLY the name of one classical recipe from {area}.
No explanation. No description. Just the recipe name.

Example output: Chiles Rellenos
"""
recipe_prompt = PromptTemplate.from_template(recipe_template)

ingredients_template =  """
provide ONLY the ingredients to make the recipe below, provide exact amounts.

Rules:
- Do NOT rewrite the recipe
- Return ONLY a bullet list
- Also provide serving size
- Be concise

Recipe:
{recipe}
"""
ingredients_prompt = PromptTemplate.from_template(ingredients_template)

instructions_template =  """
Write ONLY step-by-step instructions.

Rules:
- Do NOT repeat the recipe
- Do NOT restate ingredients
- Be concise
- Include timing and temperature

Inputs:

Recipe:
{recipe}

Ingredients:
{ingredients}
"""

instructions_prompt = PromptTemplate.from_template(instructions_template)

#generate the recipe
recipe_chain = recipe_prompt | llm
recipe = recipe_chain.invoke({"area": area}).content
#generate the ingredients
ingredients_chain = ingredients_prompt | llm
ingredients = ingredients_chain.invoke({"recipe": recipe}).content
#generate the instructions
instructions_chain = instructions_prompt | llm
instructions = instructions_chain.invoke({"recipe": recipe, "ingredients": ingredients}).content

print(f"Recipe: {recipe}")
print(f"Ingredients: {ingredients}")
print(f"Instructions: {instructions}")


history = [
    AIMessage(content=
    f"""I have generated the following recipe:
    Current area: {area}
    Current recipe: {recipe}
    Ingredients: {ingredients}
    Instructions: {instructions}
    
    If the user asks for another recipe, use area='{area}' and exclude_recipe='{recipe}'.
    """
    )
]
while True:
    question = input("Have any cooking question(time estimates, ingredient substitutions or calorie estimates): ")
    if question.lower() == "exit":
        break
    history.append(HumanMessage(content=question))

    response = llm_with_tools.invoke(history)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name == "cooking_time_estimate":
                query_recipe = tool_args["recipe"]
                result = cooking_time_estimate.invoke({"recipe": query_recipe})
                print(f"Estimated cooking time for {query_recipe}: {result} minutes")
            elif tool_name == "ingredient_subs":
                ingredient = tool_args["ingredient"]
                result = ingredient_subs.invoke({"ingredient": ingredient})
                print(f"Substitutes for {ingredient}:\n{result}")
            elif tool_name == "calorie_estimate":
                result = calorie_estimate.invoke({"recipe": recipe, "ingredients": ingredients})
                print(f"Estimated calorie count for {recipe}: {result} calories")
            elif tool_name == "serving_size":
                original_serving = tool_args["original_serving"]
                desired_serving = tool_args["desired_serving"]
                result = serving_size.invoke({"recipe": recipe, "original_serving": original_serving, "desired_serving": desired_serving})
                ingredients = result
                print(f"Adjusted ingredient amounts for {desired_serving} servings of {recipe}:\n{result}")
            elif tool_name == "difficulty_level":
                result = difficulty_level.invoke({"recipe": recipe, "ingredients": ingredients, "instructions": instructions})
                print(f"Estimated difficulty level for {recipe}: {result}")
            elif tool_name == "another_recipe":
                result = another_recipe.invoke({"area": area, "exclude_recipe": recipe})
                recipe = result
                ingredients = ingredients_chain.invoke({"recipe": recipe}).content
                instructions = instructions_chain.invoke({"recipe": recipe, "ingredients": ingredients}).content
                print(f"New Recipe: {recipe}")
                print(f"Ingredients: {ingredients}")
                print(f"Instructions: {instructions}")
                history.append(AIMessage(content=f"""
                    I have generated a new recipe:
                    Recipe: {recipe}
                    Ingredients: {ingredients}
                    Instructions: {instructions}
                    """))
                continue
            history.append(AIMessage(content=str(result)))
    else:    
        print(response.content)
        history.append(AIMessage(content=response.content))

