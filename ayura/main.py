import datetime # For Outlet_Age calculation logic in prediction tool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # Import Field for better validation
import google.generativeai as genai
import os
import uvicorn
import httpx
import re
from urllib.parse import quote_plus # For URL-encoding password

from upload_data_to_supabase import ensure_seed_and_view
ensure_seed_and_view("final_dataset_with_numeric_labels.csv")


# LangChain Imports for Text-to-SQL
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool 
from langchain_core.exceptions import OutputParserException

from dotenv import load_dotenv
load_dotenv()

# Supabase/PostgreSQL Database Configuration
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

# Global for ML Model URL
DEPLOYED_ML_MODEL_URL = os.getenv("DEPLOYED_ML_MODEL_URL", "https://model-deployment-963934256033.asia-southeast1.run.app/api/predict_outlet_performance")

# Define the mapping for predicted labels to integers (matches chatbot_server)
PREDICTION_LABEL_MAPPING = {
    "Poor": 0,
    "Medium": 1,
    "Good": 2
}

class Message(BaseModel):
    message: str

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

# LangChain SQL Agent Setup
sql_agent_executor = None # Initialize as None

# Define a custom tool for calling the ML Prediction Endpoint
@tool
def predict_item_performance(item_identifier: str, outlet_identifier: str) -> dict:
    """
    Predicts the *future sales performance* (e.g., Poor, Medium, Good) and associated sales value for a specific item at a given outlet.
    This tool is used when the user explicitly asks for a **prediction** for an item and an outlet.
    It requires the 'item_identifier' (e.g., 'FDX07') and 'outlet_identifier' (e.g., 'OUT027').
    It will internally fetch other necessary item and outlet details from the database.

    Args:
        item_identifier: The unique identifier of the item (e.g., 'FDX07').
        outlet_identifier: The unique identifier of the outlet (e.g., 'OUT027').

    Returns:
        A dictionary containing the prediction result, or an error message.
    """
    print(f"DEBUG (main.py): Tool 'predict_item_performance' called with item: {item_identifier}, outlet: {outlet_identifier}")

    # Fetch all necessary features for the ML model from the database.
    # This involves making calls to the chatbot_server's endpoints to get item and outlet details.
    try:
        # Fetch Item details
        item_response = httpx.get(f"http://localhost:8001/item_info/{item_identifier}")
        item_response.raise_for_status()
        item_data = item_response.json()
        print(f"DEBUG (main.py): Raw item_response.text: {item_response.text}") 
        print(f"DEBUG (main.py): Type of item_data after json(): {type(item_data)}")
        print(f"DEBUG (main.py): Content of item_data: {item_data}")

        # Fetch Outlet details
        outlet_response = httpx.get(f"http://localhost:8001/outlet_performance/{outlet_identifier}")
        outlet_response.raise_for_status()
        outlet_data = outlet_response.json()
        print(f"DEBUG (main.py): Raw outlet_response.text: {outlet_response.text}") 
        print(f"DEBUG (main.py): Type of outlet_data after json(): {type(outlet_data)}")
        print(f"DEBUG (main.py): Content of outlet_data: {outlet_data}")

        # Calculate Outlet_Age from Outlet_Establishment_Year
        current_year = datetime.datetime.now().year
        outlet_est_year = outlet_data.get('outlet_establishment_year', current_year)
        calculated_outlet_age = current_year - outlet_est_year

        # Construct the payload for the ML model
        payload = {
            "Item_Identifier": item_data.get("item_identifier"),
            "Item_Weight": float(item_data.get("item_weight", 0.0)), # Default to 0.0 if missing or invalid
            "Item_Fat_Content": item_data.get("item_fat_content"),
            "Item_Visibility": float(item_data.get("item_visibility", 0.0)), # Default to 0.0
            "Item_Type": item_data.get("item_type"),
            "Item_MRP": float(item_data.get("item_mrp", 0.0)), # Default to 0.0
            "item_avg_sales": float(item_data.get("item_avg_sales", 0.0)),
            "Outlet_Identifier": outlet_data.get("outlet_identifier"),
            "Outlet_Establishment_Year": outlet_est_year, # Use the fetched/defaulted establishment year
            "Outlet_Size": outlet_data.get("outlet_size"),
            "Outlet_Location_Type": outlet_data.get("outlet_location_type"),
            "Outlet_Type": outlet_data.get("outlet_type"),
        }

        # Call the new endpoint in chatbot_server.py to get and store prediction
        response = httpx.post(f"http://localhost:8001/predict_item_sales", json=payload, timeout=60.0)
        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get("detail", "Unknown error from tool server")
        print(f"ERROR (main.py): HTTP error calling prediction tool: {e.response.text}")
        return {"error": f"Failed to get prediction from tool: {error_detail}"}
    except httpx.RequestError as e:
        print(f"ERROR (main.py): Request error calling prediction tool: {str(e)}")
        return {"error": f"Failed to connect to prediction tool: {str(e)}"}
    except Exception as e:
        print(f"ERROR (main.py): Unexpected error in prediction tool: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}


try:
    # Initialize SQLDatabase connection for PostgreSQL (Supabase)
    encoded_password = quote_plus(DB_PASSWORD)
    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    print("DEBUG (main.py): Successfully initialized SQLDatabase connection for LangChain (PostgreSQL).")

    # Initialize the LLM for the agent
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.environ["GEMINI_API_KEY"])

    # Initialize Conversation Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    print("DEBUG (main.py): Conversational Memory initialized for LangChain Agent.")

    # Create the SQL Database Toolkit AND provide the custom ML prediction tool
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Pass ALL tools to create_sql_agent, including toolkit's tools and any custom tools.
    all_tools = toolkit.get_tools() + [predict_item_performance]

    # Dynamically generate tool_descriptions for the prefix
    # Instead of {tool_names}, manually list the tools with descriptions.
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in all_tools])

    # Create the SQL Agent Executor
    sql_agent_prefix = f"""
    You are an intelligent data analyst assistant. Your primary goal is to provide concise, helpful, and accurate answers based on the provided data.
    You interact with a SQL database and a machine learning model via specialized tools.

    **Decision Process:**
    1.  **Analyze User Intent**: Determine if the user is asking for:
        * **Existing Data Retrieval**: Questions about current/past facts, summaries, trends, or specific entity details (e.g., "What are the total sales?", "Show me item FDR25's type", "Worst performing item"). For these, use the SQL database tools.
        * **Future Prediction**: Questions explicitly asking for a "prediction" of sales performance (e.g., "Predict sales for FDX07 at OUT027", "What will be the performance of X?"). For these, use the `predict_item_performance` tool.
        * **Suggestions/Improvements**: Questions asking "how to improve" or "suggestions for better performance". Use common sense based on data, and general business logic.

    2.  **Contextual Understanding**: Always leverage the **entire conversation history** to resolve ambiguous references (like "it", "that outlet", "that item").
        * **Crucial Instruction**: If a user asks about an "item" or "outlet" without specifying an identifier, but an identifier (e.g., 'OUT027', 'FDR25') was clearly mentioned in *any previous turn in the conversation*, assume they are referring to that most recently discussed entity. Explicitly use this identifier when forming SQL queries or tool arguments.
        * **Example for LLM**: If the previous AI response mentioned "outlet OUT027" or "item FDR25", and the user then says "that outlet" or "that item", you **MUST** substitute the full identifier ('OUT027' or 'FDR25') into your next action/tool call.

    **Tool Usage Guidelines:**
    * **SQL Database Tools**: Use these for all data retrieval, aggregation, and lookup tasks. When performing SQL queries, ONLY output the raw SQL query. DO NOT wrap the SQL in markdown code blocks (```sql).
    * **`predict_item_performance` Tool**: This tool is for GENERATING NEW PREDICTIONS. You MUST use this tool when the user explicitly asks for a "prediction". It requires both `item_identifier` and `outlet_identifier` as arguments. You must ensure these are identified from the current query or conversation history before calling this tool.

    **Response Formatting:**
    * Do NOT ask clarifying questions if you can infer from the conversation or use your tools.
    * Always aim to provide a direct answer or execute a relevant tool based on available context.
    * **CRITICAL**: If you genuinely cannot fulfill the request after using all your capabilities and considering context, your final output MUST be `Final Answer: I cannot answer this question based on the available data.` or a similar helpful message formatted with `Final Answer:`.
    * When you have a final answer, it MUST start with "Final Answer:".

    You have access to the following tools:
    {tool_descriptions}

    Example conversation for context resolution:
    User: What is the worst performing item in OUT027?
    AI: Final Answer: The item with identifier FDR25 has the worst performance at outlet OUT027, with an item sales deviation of -2388.26.
    User: What is its item_type?
    AI: Final Answer: The item type for FDR25 is 'Dairy'.
    User: Could you remind me what I asked earlier?
    AI: Final Answer: You asked about the worst performing item in OUT027 and its item type.
    """

    sql_agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit, # Still pass the toolkit for its internal setup
        tools=all_tools, # Pass the combined list of tools
        verbose=True,
        handle_parsing_errors=True, # Try to retry if can't parse intermediate steps
        agent_executor_kwargs={"memory": memory},
        prefix=sql_agent_prefix # Pass the dynamically formatted prefix
    )
    print("DEBUG (main.py): LangChain SQL Agent initialized with memory, enhanced prefix, and ML prediction tool.")

except Exception as e:
    print(f"CRITICAL ERROR (main.py): Failed to initialize LangChain SQL Agent: {e}")

# Initialize a separate Gemini model for general chat (for initial greetings)
general_chat_model = genai.GenerativeModel(model_name="gemini-1.5-flash")


@app.post("/chat")
async def chat(message: Message):
    user_query = message.message
    response_text = "I'm sorry, I couldn't process that request."

    try:
        # Check for simple greetings first
        if any(greeting in user_query.lower() for greeting in ["hi", "hello", "hey", "how are you"]):
            chat_session = general_chat_model.start_chat()
            gemini_response = chat_session.send_message(message.message)
            if gemini_response.text:
                response_text = gemini_response.text
            else:
                response_text = "Hello there! How can I help you today?"
            return {"response": response_text}

        # Use LangChain SQL Agent for Data-related Queries
        if sql_agent_executor:
            print(f"DEBUG (main.py): Sending query to SQL Agent: '{user_query}'")

            try:
                agent_raw_response = sql_agent_executor.run(user_query)
                response_text = agent_raw_response
                print(f"DEBUG (main.py): SQL Agent final response: {response_text}")
            except OutputParserException as e:
                print(f"DEBUG (main.py): OutputParsingError caught: {e}")
                # Analyze the error message to provide context-specific feedback
                if "I cannot answer this question based on the available data." in str(e):
                    response_text = "I'm sorry, I cannot answer this question based on the information I have available. Perhaps I lack the specific data or a clear understanding of your request. Could you rephrase or provide more details?"
                elif "Final Answer:" not in str(e): # General parsing error if no Final Answer
                    response_text = "I'm sorry, I had trouble understanding my own internal steps to get to an answer. Please try again or rephrase your question."
                else: # Fallback for other OutputParserException if it somehow contains Final Answer
                    response_text = f"I encountered an internal error: {str(e)}"
            except Exception as e:
                print(f"CRITICAL ERROR (main.py): An unexpected error occurred during agent execution: {str(e)}")
                # Provide a more general user-friendly message for other unexpected errors
                response_text = "I encountered an internal error trying to fulfill your request. Please check the server logs for more details."
        else:
            response_text = "Database query functionality is not available due to a setup error. Please check server logs."
            print(f"DEBUG (main.py): SQL Agent not initialized.")

        return {"response": response_text}

    except Exception as e: # This outer catch is for errors BEFORE agent execution, or broad app errors
        print(f"CRITICAL ERROR (main.py): An unexpected error occurred in /chat: {str(e)}")
        return {"response": f"I encountered an unexpected internal error trying to fulfill your request: {str(e)}. Please check server logs for more details."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)