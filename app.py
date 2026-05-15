import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

@tool
def another_recipe(area: str, exclude_recipe: str) -> str:
    """Call this tool whenever the user asks for a new recipe, different recipe, or another recipe. 
    ALWAYS use the area and current recipe from the conversation history. 
    NEVER ask the user for area or exclude_recipe."""
    prompt = f"""You are a culinary expert.
    Return ONLY the name of one classical recipe from {area} that is NOT {exclude_recipe}.
    No explanation. No description. Just the recipe name.
    Example output: Chiles Rellenos"""
    return llm.invoke(prompt).content

@tool
def difficulty_level(recipe: str, ingredients: str, instructions: str) -> str:
    """This tool takes a recipe name, its ingredients and instructions and returns an estimated difficulty level."""
    prompt = f"""You are a culinary expert.
    Estimate the difficulty level of the following recipe: {recipe}
    ingredients: {ingredients}
    instructions: {instructions}
    Rules:
    - Return only Easy, Medium, or Hard
    - Follow with one sentence explaining why
    - Be concise"""
    return llm.invoke(prompt).content

@tool
def serving_size(recipe: str, original_serving: str, desired_serving: str) -> str:
    """Call this tool whenever the user asks to change the serving size, portions, or amounts.
    ALWAYS use the recipe and original serving from the conversation history.
    NEVER ask the user for recipe or original serving."""
    prompt = f"""You are a culinary expert.
    Adjust the ingredient amounts for {recipe} from {original_serving} servings to {desired_serving} servings.
    Rules:
    - Return only the adjusted ingredient list with exact amounts
    - No explanation
    - Be concise
    - Return in bullet list format"""
    return llm.invoke(prompt).content

@tool
def calorie_estimate(recipe: str, ingredients: str) -> str:
    """Call this tool whenever the user asks about calories, calorie count, or how many calories.
    ALWAYS use the recipe and ingredients from the conversation history.
    NEVER ask the user for recipe or ingredients."""
    prompt = f"""You are a nutritional expert.
    Estimate the calorie count for the following recipe: {recipe}
    ingredients: {ingredients}
    Rules:
    - Return only the estimated calorie count
    - No explanation
    - Be concise"""
    return llm.invoke(prompt).content

@tool
def cooking_time_estimate(recipe: str) -> str:
    """This tool takes a recipe name and returns an estimated cooking time."""
    prompt = f"""You are a culinary expert.
    Estimate the cooking time for: {recipe}
    This could be a full recipe name or a specific cooking step.
    Rules:
    - Return only the estimated time in minutes
    - No explanation
    - Be concise"""
    return llm.invoke(prompt).content

@tool
def ingredient_subs(ingredient: str) -> str:
    """This tool takes an ingredient and returns 2 cooking substitutes."""
    prompt = f"""You are a cooking ingredient substitution expert.
    Give 2 realistic substitutes for {ingredient} used in cooking.
    Rules:
    - Return only ingredient names
    - No numbering
    - No explanation
    - One substitute per line
    - Do NOT include the original ingredient
    - Substitutes must serve the same cooking purpose"""
    return llm.invoke(prompt).content

llm_with_tools = llm.bind_tools([ingredient_subs, cooking_time_estimate, calorie_estimate, serving_size, difficulty_level, another_recipe])

recipe_prompt = PromptTemplate.from_template("""You are a culinary expert. 
Return ONLY the name of one classical recipe from {area}.
No explanation. No description. Just the recipe name.
Example output: Chiles Rellenos""")

ingredients_prompt = PromptTemplate.from_template("""
provide ONLY the ingredients to make the recipe below, provide exact amounts.
Rules:
- Do NOT rewrite the recipe
- Return ONLY a bullet list
- Also provide serving size
- Be concise
Recipe:
{recipe}""")

instructions_prompt = PromptTemplate.from_template("""
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
{ingredients}""")

recipe_chain = recipe_prompt | llm
ingredients_chain = ingredients_prompt | llm
instructions_chain = instructions_prompt | llm

# UI
st.title("World Recipe Generator")
st.write("Enter an area of the world to get a classical recipe from that region.")

area = st.text_input("Enter the area of the world you want a recipe from:")

if st.button("Generate Recipe"):
    with st.spinner("Generating recipe..."):
        st.session_state.area = area
        st.session_state.recipe = recipe_chain.invoke({"area": area}).content
        st.session_state.ingredients = ingredients_chain.invoke({"recipe": st.session_state.recipe}).content
        st.session_state.instructions = instructions_chain.invoke({"recipe": st.session_state.recipe, "ingredients": st.session_state.ingredients}).content
        st.session_state.history = [
            AIMessage(content=f"""I have generated the following recipe:
            Current area: {area}
            Current recipe: {st.session_state.recipe}
            Ingredients: {st.session_state.ingredients}
            Instructions: {st.session_state.instructions}
            If the user asks for another recipe, use area='{area}' and exclude_recipe='{st.session_state.recipe}'.
            """)
        ]
        st.session_state.messages = []

if "recipe" in st.session_state:
    st.subheader(f"Recipe: {st.session_state.recipe}")
    st.markdown(st.session_state.ingredients)
    st.markdown(st.session_state.instructions)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a cooking question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.history.append(HumanMessage(content=question))
        response = llm_with_tools.invoke(st.session_state.history)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name == "cooking_time_estimate":
                    query_recipe = tool_args["recipe"]
                    result = cooking_time_estimate.invoke({"recipe": query_recipe})
                    answer = f"Estimated cooking time for {query_recipe}: {result} minutes"
                elif tool_name == "ingredient_subs":
                    ingredient = tool_args["ingredient"]
                    result = ingredient_subs.invoke({"ingredient": ingredient})
                    answer = f"Substitutes for {ingredient}:\n{result}"
                elif tool_name == "calorie_estimate":
                    result = calorie_estimate.invoke({"recipe": st.session_state.recipe, "ingredients": st.session_state.ingredients})
                    answer = f"Estimated calorie count for {st.session_state.recipe}: {result} calories"
                elif tool_name == "serving_size":
                    original_serving = tool_args.get("original_serving", "unknown")
                    desired_serving = tool_args.get("desired_serving", "unknown")
                    result = serving_size.invoke({"recipe": st.session_state.recipe, "original_serving": original_serving, "desired_serving": desired_serving})
                    st.session_state.ingredients = result
                    answer = f"Adjusted ingredients for {desired_serving} servings of {st.session_state.recipe}:\n{result}"
                    st.session_state.history.append(AIMessage(content=answer))
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
                elif tool_name == "difficulty_level":
                    result = difficulty_level.invoke({"recipe": st.session_state.recipe, "ingredients": st.session_state.ingredients, "instructions": st.session_state.instructions})
                    answer = f"Difficulty level for {st.session_state.recipe}: {result}"
                elif tool_name == "another_recipe":
                    result = another_recipe.invoke({"area": st.session_state.area, "exclude_recipe": st.session_state.recipe})
                    st.session_state.recipe = result
                    st.session_state.ingredients = ingredients_chain.invoke({"recipe": st.session_state.recipe}).content
                    st.session_state.instructions = instructions_chain.invoke({"recipe": st.session_state.recipe, "ingredients": st.session_state.ingredients}).content
                    answer = f"New Recipe: {st.session_state.recipe}\n\n{st.session_state.ingredients}\n\n{st.session_state.instructions}"
                    st.session_state.history.append(AIMessage(content=f"""
                        I have generated a new recipe:
                        Current recipe: {st.session_state.recipe}
                        Ingredients: {st.session_state.ingredients}
                        Instructions: {st.session_state.instructions}
                        If the user asks for another recipe, use area='{st.session_state.area}' and exclude_recipe='{st.session_state.recipe}'.
                    """))
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.history.append(AIMessage(content=str(result)))
        else:
            answer = response.content
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.history.append(AIMessage(content=answer))

        st.rerun()