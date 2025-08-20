import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import re
import httpx
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
import uvicorn

# Supabase
from utils.upload_data_to_supabase import ensure_seed_and_view
ensure_seed_and_view("final_dataset_with_numeric_labels.csv")

# SQLAlchemy Imports
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func as sqlalchemy_func

# LangChain Imports for Text-to-SQL
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool 
from langchain_core.exceptions import OutputParserException

# Google Generative AI
import google.generativeai as genai

load_dotenv()

# =============================================================================
# Database Configuration & Models
# =============================================================================

# Supabase/PostgreSQL Credentials
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

# URL-encode the password for the database URI
ENCODED_DB_PASSWORD = quote_plus(DB_PASSWORD)

# SQLAlchemy Database URL
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{ENCODED_DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy Engine and Session setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Models
class Item(Base):
    __tablename__ = 'items'
    item_id = Column(Integer, primary_key=True, index=True)
    item_identifier = Column(String, nullable=False, index=True)
    item_weight = Column(Float)
    item_fat_content = Column(String)
    item_visibility = Column(Float)
    item_type = Column(String)
    item_mrp = Column(Float)
    item_avg_sales = Column(Float)

class Outlet(Base):
    __tablename__ = 'outlets'
    outlet_id = Column(Integer, primary_key=True, index=True)
    outlet_identifier = Column(String, nullable=False, index=True)
    outlet_size = Column(String)
    outlet_location_type = Column(String)
    outlet_type = Column(String)
    outlet_age = Column(Integer)
    outlet_establishment_year = Column(Integer)
    item_outlet_sales = Column(Float)
    item_sales_deviation = Column(Float)

class PerformancePrediction(Base):
    __tablename__ = 'performance_predictions'
    prediction_id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey('items.item_id'), nullable=False)
    outlet_id = Column(Integer, ForeignKey('outlets.outlet_id'), nullable=False)
    item_sales_deviation = Column(Float)
    performance_label = Column(Integer)
    predicted_label = Column(Integer)

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    print("DEBUG: Database tables checked/created via SQLAlchemy.")
except SQLAlchemyError as e:
    print(f"CRITICAL ERROR: Failed to create database tables: {e}")
    raise SystemExit("Database table creation failed. Exiting.")

# FastAPI Dependency to get DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# ML Model Configuration
# =============================================================================

DEPLOYED_ML_MODEL_URL = os.getenv(
    "DEPLOYED_ML_MODEL_URL",
    "https://model-deployment-963934256033.asia-southeast1.run.app/api/predict_outlet_performance"
)
PREDICTION_LABEL_MAPPING = {"Poor": 0, "Medium": 1, "Good": 2}

# =============================================================================
# Pydantic Models
# =============================================================================

class Message(BaseModel):
    message: str

class SalesQuery(BaseModel):
    query: str

class PredictionInput(BaseModel):
    # Item features
    Item_Identifier: str = Field(..., description="Unique ID of the item (e.g., FDX07)")
    Item_Weight: float = Field(..., description="Weight of the item")
    Item_Fat_Content: str = Field(..., description="Fat content (e.g., 'Low Fat', 'Regular', 'LF')")
    Item_Visibility: float = Field(..., description="Visibility of the item in the outlet")
    Item_Type: str = Field(..., description="Category of the item (e.g., 'Dairy', 'Snack Foods')")
    Item_MRP: float = Field(..., description="Maximum Retail Price of the item")

    # Outlet features
    Outlet_Identifier: str = Field(..., description="Unique ID of the outlet (e.g., OUT027)")
    Outlet_Establishment_Year: int = Field(..., description="Year the outlet was established")
    Outlet_Size: str = Field(..., description="Size of the outlet (e.g., 'Medium', 'Small', 'High')")
    Outlet_Location_Type: str = Field(..., description="Type of location (e.g., 'Tier 1', 'Tier 2', 'Tier 3')")
    Outlet_Type: str = Field(..., description="Type of outlet (e.g., 'Supermarket Type1', 'Grocery Store')")

    # Observation data
    item_outlet_sales: float = Field(..., description="Observed sales for this item at this outlet")
    item_sales_deviation: float | None = Field(
        default=None,
        description="If provided, used directly; otherwise computed as item_outlet_sales - historical average"
    )
    always_add_new_rows: bool = Field(
        default=True,
        description="When True, inserts new rows into items/outlets/performance_predictions (append-only)"
    )

# =============================================================================
# LangChain Setup
# =============================================================================

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Define custom tool for ML predictions
@tool
def predict_item_performance(item_identifier: str, outlet_identifier: str) -> dict:
    """
    Predicts the future sales performance (e.g., Poor, Medium, Good) for a specific item at a given outlet.
    This tool is used when the user explicitly asks for a prediction.
    
    Args:
        item_identifier: The unique identifier of the item (e.g., 'FDX07').
        outlet_identifier: The unique identifier of the outlet (e.g., 'OUT027').
    
    Returns:
        A dictionary containing the prediction result, or an error message.
    """
    print(f"DEBUG: Tool 'predict_item_performance' called with item: {item_identifier}, outlet: {outlet_identifier}")

    try:
        # Use internal function to get item and outlet data
        from fastapi import Request
        
        # Create a mock request object for dependency injection
        class MockRequest:
            pass
        
        # Get database session
        db_gen = get_db()
        db = next(db_gen)
        
        try:
            # Fetch item data
            item = db.query(Item).filter(Item.item_identifier == item_identifier).first()
            if not item:
                return {"error": f"Item {item_identifier} not found"}
            
            # Fetch outlet data
            outlet = db.query(Outlet).filter(Outlet.outlet_identifier == outlet_identifier).first()
            if not outlet:
                return {"error": f"Outlet {outlet_identifier} not found"}
            
            # Construct payload for ML model
            payload = {
                "Item_Identifier": item.item_identifier,
                "Item_Weight": item.item_weight,
                "Item_Fat_Content": item.item_fat_content,
                "Item_Visibility": item.item_visibility,
                "Item_Type": item.item_type,
                "Item_MRP": item.item_mrp,
                "Outlet_Identifier": outlet.outlet_identifier,
                "Outlet_Establishment_Year": outlet.outlet_establishment_year,
                "Outlet_Size": outlet.outlet_size,
                "Outlet_Location_Type": outlet.outlet_location_type,
                "Outlet_Type": outlet.outlet_type
            }
            
            # Call ML model
            response = httpx.post(DEPLOYED_ML_MODEL_URL, json=payload, timeout=60.0)
            response.raise_for_status()
            return response.json()
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"ERROR: Prediction tool error: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

# Initialize LangChain components
sql_agent_executor = None
general_chat_model = None

try:
    # Initialize SQLDatabase connection
    encoded_password = quote_plus(DB_PASSWORD)
    langchain_db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    print("DEBUG: Successfully initialized SQLDatabase connection for LangChain.")

    # Initialize LLM for the agent
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.environ["GEMINI_API_KEY"])

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    print("DEBUG: Conversational Memory initialized for LangChain Agent.")

    # Create SQL Database Toolkit and add custom tools
    toolkit = SQLDatabaseToolkit(db=langchain_db, llm=llm)
    all_tools = toolkit.get_tools() + [predict_item_performance]

    # Create tool descriptions
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
        toolkit=toolkit,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True,
        agent_executor_kwargs={"memory": memory},
        prefix=sql_agent_prefix
    )
    print("DEBUG: LangChain SQL Agent initialized successfully.")

    # Initialize general chat model
    general_chat_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize LangChain components: {e}")

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title="Unified Sales Analytics API", version="1.0.0")

# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get("/test")
def test_endpoint():
    print("DEBUG: /test endpoint HIT!")
    return {"message": "Unified server is running and responding!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

# =============================================================================
# Data Retrieval Endpoints
# =============================================================================

@app.post("/analyze_sales_data")
def analyze_sales_data(sales_query: SalesQuery, db: Session = Depends(get_db)):
    query = sales_query.query.lower()
    print(f"DEBUG: analyze_sales_data endpoint HIT with query: '{query}'")

    try:
        if "total item sales" in query:
            total_item_sales = db.query(sqlalchemy_func.sum(Outlet.item_outlet_sales)).scalar()
            return {"result": f"Total Item Sales across all outlets: {total_item_sales}"}

        elif "average item sales" in query:
            average_item_sales = db.query(sqlalchemy_func.avg(Outlet.item_outlet_sales)).scalar()
            return {"result": f"Average Item Sales across all outlets: {average_item_sales}"}

        match = re.search(r'out\d{3}', query)
        if match:
            outlet_identifier = match.group(0).upper()
            outlet_sales = db.query(sqlalchemy_func.avg(Outlet.item_outlet_sales))\
                             .filter(Outlet.outlet_identifier == outlet_identifier)\
                             .scalar()
            if outlet_sales is not None:
                return {"result": f"Average item sales for {outlet_identifier}: {outlet_sales}"}
            else:
                return {"result": f"No item sales data found for {outlet_identifier} or outlet not found."}
        else:
            return {"error": "Query not recognized. Ask about total item sales, average item sales, or item sales for a specific outlet (e.g., 'OUT049')."}

    except SQLAlchemyError as err:
        print(f"ERROR: SQLAlchemy error in analyze_sales_data: {err}")
        raise HTTPException(status_code=500, detail=f"Database error analyzing sales data: {err}")
    except Exception as e:
        print(f"ERROR: Unexpected error in analyze_sales_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while analyzing sales data: {str(e)}")

@app.get("/item_info/{item_identifier}")
def get_item_info_by_identifier(item_identifier: str, db: Session = Depends(get_db)) -> dict:
    try:
        print(f"DEBUG: get_item_info: Attempting to query for item_identifier: {item_identifier}")
        item = db.query(Item).filter(Item.item_identifier == item_identifier).first()
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        
        item_data = {
            "item_id": item.item_id,
            "item_identifier": item.item_identifier,
            "item_weight": item.item_weight,
            "item_fat_content": item.item_fat_content,
            "item_visibility": item.item_visibility,
            "item_type": item.item_type,
            "item_mrp": item.item_mrp,
            "item_avg_sales": item.item_avg_sales
        }
        print(f"DEBUG: get_item_info: DB result for {item_identifier}: {item_data}")
        return item_data
    except SQLAlchemyError as err:
        print(f"ERROR: SQLAlchemy error in get_item_info_by_identifier: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching item info: {err}")
    except Exception as e:
        print(f"ERROR: Unexpected error in get_item_info_by_identifier: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching item data: {str(e)}")

@app.get("/outlets_list")
def get_outlets_list(db: Session = Depends(get_db)) -> list[str]:
    try:
        outlets = db.query(Outlet.outlet_identifier).distinct().order_by(Outlet.outlet_identifier).all()
        results = [o[0] for o in outlets]
        print(f"DEBUG: Fetched {len(results)} distinct outlet identifiers.")
        return results
    except SQLAlchemyError as err:
        print(f"ERROR: SQLAlchemy error in get_outlets_list: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching outlet list: {err}")
    except Exception as e:
        print(f"ERROR: Unexpected error in get_outlets_list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching outlet list: {str(e)}")

@app.get("/items_list")
def get_items_list(db: Session = Depends(get_db)) -> list[str]:
    try:
        items = db.query(Item.item_identifier).distinct().order_by(Item.item_identifier).all()
        results = [item[0] for item in items]
        print(f"DEBUG: Fetched {len(results)} distinct item identifiers.")
        return results
    except SQLAlchemyError as err:
        print(f"ERROR: SQLAlchemy error in get_items_list: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching item list: {err}")
    except Exception as e:
        print(f"ERROR: Unexpected error in get_items_list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching item list: {str(e)}")

@app.get("/outlet_performance/{outlet_identifier}")
def get_outlet_performance_by_identifier(outlet_identifier: str, db: Session = Depends(get_db)) -> dict:
    try:
        print(f"DEBUG: get_outlet_performance: Attempting to query for outlet_identifier: {outlet_identifier}")
        outlet = db.query(Outlet).filter(Outlet.outlet_identifier == outlet_identifier).first()
        if not outlet:
            raise HTTPException(status_code=404, detail="Outlet not found")
        
        outlet_data = {
            "outlet_id": outlet.outlet_id,
            "outlet_identifier": outlet.outlet_identifier,
            "outlet_size": outlet.outlet_size,
            "outlet_type": outlet.outlet_type,
            "outlet_location_type": outlet.outlet_location_type,
            "outlet_age": outlet.outlet_age,
            "outlet_establishment_year": outlet.outlet_establishment_year,
            "item_outlet_sales": outlet.item_outlet_sales,
            "item_sales_deviation": outlet.item_sales_deviation
        }
        print(f"DEBUG: get_outlet_performance: DB result for {outlet_identifier}: {outlet_data}")
        return outlet_data
    except SQLAlchemyError as err:
        print(f"ERROR: SQLAlchemy error in get_outlet_performance_by_identifier: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching outlet performance: {err}")
    except Exception as e:
        print(f"ERROR: Unexpected error in get_outlet_performance_by_identifier: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching outlet data: {str(e)}")

@app.get("/prediction_info/{item_identifier}/{outlet_identifier}")
def get_prediction_info(item_identifier: str, outlet_identifier: str, db: Session = Depends(get_db)) -> dict:
    try:
        print(f"DEBUG: get_prediction_info: Attempting to query for item: {item_identifier}, outlet: {outlet_identifier}")
        prediction = (
            db.query(PerformancePrediction)
              .join(Item, Item.item_id == PerformancePrediction.item_id)
              .join(Outlet, Outlet.outlet_id == PerformancePrediction.outlet_id)
              .filter(Item.item_identifier == item_identifier,
                      Outlet.outlet_identifier == outlet_identifier)
              .order_by(PerformancePrediction.prediction_id.desc())
              .first()
        )
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found for this item and outlet")
        
        prediction_data = {
            "prediction_id": prediction.prediction_id,
            "item_sales_deviation": prediction.item_sales_deviation,
            "performance_label": prediction.performance_label,
            "predicted_label": prediction.predicted_label
        }
        print(f"DEBUG: get_prediction_info: DB result for {item_identifier}-{outlet_identifier}: {prediction_data}")
        return prediction_data
    except SQLAlchemyError as err:
        print(f"ERROR: SQLAlchemy error in get_prediction_info: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching prediction info: {err}")
    except Exception as e:
        print(f"ERROR: Unexpected error in get_prediction_info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching prediction data: {str(e)}")

# =============================================================================
# ML Prediction Endpoints
# =============================================================================

@app.post("/predict_item_sales")
async def predict_item_sales(prediction_input: PredictionInput, db: Session = Depends(get_db)):
    print(f"DEBUG: /predict_item_sales input: {prediction_input.dict()}")

    if not DEPLOYED_ML_MODEL_URL:
        raise HTTPException(status_code=500, detail="Deployed ML Model URL is not configured.")

    # Compute historical average & deviation
    observed_sales = float(prediction_input.item_outlet_sales)

    # Find historical data for this item
    existing_item_ids = db.query(Item.item_id)\
                          .filter(Item.item_identifier == prediction_input.Item_Identifier)\
                          .all()
    existing_item_ids = [row[0] for row in existing_item_ids]

    past_avg = None
    if existing_item_ids:
        past_avg = db.query(sqlalchemy_func.avg(Outlet.item_outlet_sales))\
                     .join(PerformancePrediction, PerformancePrediction.outlet_id == Outlet.outlet_id)\
                     .filter(
                         PerformancePrediction.item_id.in_(existing_item_ids),
                         Outlet.item_outlet_sales.isnot(None)
                     )\
                     .scalar()

    computed_item_avg_sales = float(past_avg) if past_avg is not None else observed_sales
    computed_item_sales_deviation = (
        float(prediction_input.item_sales_deviation)
        if prediction_input.item_sales_deviation is not None
        else observed_sales - computed_item_avg_sales
    )

    print(f"DEBUG: computed historical avg={computed_item_avg_sales:.4f}, "
          f"observed={observed_sales:.2f}, deviation={computed_item_sales_deviation:.4f}")

    # Call ML model
    ml_payload = {
        "Item_Identifier": prediction_input.Item_Identifier,
        "Item_Weight": prediction_input.Item_Weight,
        "Item_Fat_Content": prediction_input.Item_Fat_Content,
        "Item_Visibility": prediction_input.Item_Visibility,
        "Item_Type": prediction_input.Item_Type,
        "Item_MRP": prediction_input.Item_MRP,
        "Outlet_Identifier": prediction_input.Outlet_Identifier,
        "Outlet_Establishment_Year": prediction_input.Outlet_Establishment_Year,
        "Outlet_Size": prediction_input.Outlet_Size,
        "Outlet_Location_Type": prediction_input.Outlet_Location_Type,
        "Outlet_Type": prediction_input.Outlet_Type
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(DEPLOYED_ML_MODEL_URL, json=ml_payload, timeout=60.0)
            response.raise_for_status()
        prediction_result = response.json()
        print(f"DEBUG: Raw Prediction from ML Model: {prediction_result}")
    except httpx.RequestError as exc:
        print(f"ERROR: ML request error: {exc}")
        raise HTTPException(status_code=503, detail=f"Could not connect to deployed ML model: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"ERROR: ML bad status {exc.response.status_code}: {exc.response.text}")
        raise HTTPException(status_code=exc.response.status_code, detail=f"ML Model API error: {exc.response.text}")

    predicted_string_label = prediction_result.get("predicted_outlet_performance", "Unknown")
    predicted_label_int = PREDICTION_LABEL_MAPPING.get(predicted_string_label, 99)
    predicted_sales_value = prediction_result.get("predicted_sales_value", None)
    
    try:
        predicted_sales_value = float(predicted_sales_value) if predicted_sales_value is not None else None
    except ValueError:
        predicted_sales_value = None

    # Store prediction in database (append-only)
    try:
        # Insert new Item row
        new_item = Item(
            item_identifier=prediction_input.Item_Identifier,
            item_weight=prediction_input.Item_Weight,
            item_fat_content=prediction_input.Item_Fat_Content,
            item_visibility=prediction_input.Item_Visibility,
            item_type=prediction_input.Item_Type,
            item_mrp=prediction_input.Item_MRP,
            item_avg_sales=computed_item_avg_sales
        )
        db.add(new_item)
        db.flush()
        item_id = new_item.item_id
        print(f"DEBUG: Inserted new item row id={item_id}")

        # Insert new Outlet row
        current_year = datetime.datetime.now().year
        calculated_outlet_age = current_year - prediction_input.Outlet_Establishment_Year
        new_outlet = Outlet(
            outlet_identifier=prediction_input.Outlet_Identifier,
            outlet_size=prediction_input.Outlet_Size,
            outlet_location_type=prediction_input.Outlet_Location_Type,
            outlet_type=prediction_input.Outlet_Type,
            outlet_age=calculated_outlet_age,
            outlet_establishment_year=prediction_input.Outlet_Establishment_Year,
            item_outlet_sales=observed_sales,
            item_sales_deviation=computed_item_sales_deviation
        )
        db.add(new_outlet)
        db.flush()
        outlet_id = new_outlet.outlet_id
        print(f"DEBUG: Inserted new outlet row id={outlet_id}")

        # Insert new PerformancePrediction row
        new_prediction = PerformancePrediction(
            item_id=item_id,
            outlet_id=outlet_id,
            item_sales_deviation=computed_item_sales_deviation,
            performance_label=predicted_label_int,
            predicted_label=predicted_label_int
        )
        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)
        print(f"DEBUG: Inserted performance_predictions id={new_prediction.prediction_id}")

        storage_status = "Appended new rows to items, outlets, and performance_predictions."

    except SQLAlchemyError as err:
        print(f"ERROR: SQLAlchemy storage error: {err}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database storage failed: {err}")
    except Exception as e:
        print(f"ERROR: Unexpected error during storage: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Unexpected storage error: {e}")

    return {
        "prediction_label": predicted_string_label,
        "predicted_sales_value": predicted_sales_value,
        "observed_item_outlet_sales": observed_sales,
        "computed_item_avg_sales": computed_item_avg_sales,
        "computed_item_sales_deviation": computed_item_sales_deviation,
        "storage_status": storage_status
    }

# =============================================================================
# Chat Endpoints
# =============================================================================

@app.post("/chat")
async def chat(message: Message):
    user_query = message.message
    response_text = "I'm sorry, I couldn't process that request."

    try:
        # Check for simple greetings first
        if any(greeting in user_query.lower() for greeting in ["hi", "hello", "hey", "how are you"]):
            if general_chat_model:
                chat_session = general_chat_model.start_chat()
                gemini_response = chat_session.send_message(message.message)
                response_text = gemini_response.text if gemini_response.text else "Hello there! How can I help you today?"
            else:
                response_text = "Hello there! How can I help you today?"
            return {"response": response_text}

        # Use LangChain SQL Agent for data-related queries
        if sql_agent_executor:
            print(f"DEBUG: Sending query to SQL Agent: '{user_query}'")

            try:
                agent_raw_response = sql_agent_executor.run(user_query)
                response_text = agent_raw_response
                print(f"DEBUG: SQL Agent final response: {response_text}")
            except OutputParserException as e:
                print(f"DEBUG: OutputParsingError caught: {e}")
                if "I cannot answer this question based on the available data." in str(e):
                    response_text = "I'm sorry, I cannot answer this question based on the information I have available. Perhaps I lack the specific data or a clear understanding of your request. Could you rephrase or provide more details?"
                elif "Final Answer:" not in str(e):
                    response_text = "I'm sorry, I had trouble understanding my own internal steps to get to an answer. Please try again or rephrase your question."
                else:
                    response_text = f"I encountered an internal error: {str(e)}"
            except Exception as e:
                print(f"CRITICAL ERROR: An unexpected error occurred during agent execution: {str(e)}")
                response_text = "I encountered an internal error trying to fulfill your request. Please check the server logs for more details."
        else:
            response_text = "Database query functionality is not available due to a setup error. Please check server logs."
            print(f"DEBUG: SQL Agent not initialized.")

        return {"response": response_text}

    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred in /chat: {str(e)}")
        return {"response": f"I encountered an unexpected internal error trying to fulfill your request: {str(e)}. Please check server logs for more details."}

# =============================================================================
# Startup and Initialization
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize data seeding and any other startup tasks"""
    try:
        # Import and run your data seeding function
        from utils.upload_data_to_supabase import ensure_seed_and_view
        ensure_seed_and_view("final_dataset_with_numeric_labels.csv")
        print("DEBUG: Data seeding completed successfully.")
    except Exception as e:
        print(f"WARNING: Data seeding failed: {e}")

# =============================================================================
# Main Server Function
# =============================================================================

def run_server():
    """Run the unified FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()