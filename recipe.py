import os
from dotenv import load_dotenv, dotenv_values
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
#choose model here
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)
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