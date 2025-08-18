import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import re
import httpx
import os
from urllib.parse import quote_plus  # For URL-encoding password
from dotenv import load_dotenv
load_dotenv()

# SQLAlchemy Imports
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func as sqlalchemy_func
import uvicorn

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
    # item_identifier is not unique (append-only design)
    item_identifier = Column(String, nullable=False, index=True)
    item_weight = Column(Float)
    item_fat_content = Column(String)
    item_visibility = Column(Float)
    item_type = Column(String)
    item_mrp = Column(Float)
    item_avg_sales = Column(Float)  # stored per observation row


class Outlet(Base):
    __tablename__ = 'outlets'
    outlet_id = Column(Integer, primary_key=True, index=True)
    # outlet_identifier is not unique (append-only design)
    outlet_identifier = Column(String, nullable=False, index=True)
    outlet_size = Column(String)
    outlet_location_type = Column(String)
    outlet_type = Column(String)
    outlet_age = Column(Integer)
    outlet_establishment_year = Column(Integer)
    # per-observation figures
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
    # NOTE: no timestamp column — sort by prediction_id when recency needed


# Create database tables (if they don't exist) - This is run on startup
try:
    Base.metadata.create_all(bind=engine)
    print("DEBUG (chatbot_server): Database tables checked/created via SQLAlchemy.")
except SQLAlchemyError as e:
    print(f"CRITICAL ERROR (chatbot_server): Failed to create database tables: {e}")
    raise SystemExit("Database table creation failed. Exiting.")

# FastAPI Dependency to get DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

# ML Model URL and Label Mapping
DEPLOYED_ML_MODEL_URL = os.getenv(
    "DEPLOYED_ML_MODEL_URL",
    "https://model-deployment-963934256033.asia-southeast1.run.app/api/predict_outlet_performance"
)
PREDICTION_LABEL_MAPPING = {"Poor": 0, "Medium": 1, "Good": 2}

@app.get("/test")
def test_endpoint():
    print("DEBUG (chatbot_server): /test endpoint HIT!")
    return {"message": "Chatbot server is running and responding!"}

class SalesQuery(BaseModel):
    query: str

@app.post("/analyze_sales_data")
def analyze_sales_data(sales_query: SalesQuery, db: Session = Depends(get_db)):
    query = sales_query.query.lower()
    print(f"DEBUG (chatbot_server): analyze_sales_data endpoint HIT with query: '{query}'")

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
        print(f"ERROR (chatbot_server): SQLAlchemy error in analyze_sales_data: {err}")
        raise HTTPException(status_code=500, detail=f"Database error analyzing sales data: {err}")
    except Exception as e:
        print(f"ERROR (chatbot_server): Unexpected error in analyze_sales_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while analyzing sales data: {str(e)}")


@app.get("/item_info/{item_identifier}")
def get_item_info_by_identifier(item_identifier: str, db: Session = Depends(get_db)) -> dict:
    try:
        print(f"DEBUG (chatbot_server): get_item_info: Attempting to query for item_identifier: {item_identifier}")
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
        print(f"DEBUG (chatbot_server): get_item_info: DB result for {item_identifier}: {item_data}")
        return item_data
    except SQLAlchemyError as err:
        print(f"ERROR (chatbot_server): SQLAlchemy error in get_item_info_by_identifier: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching item info: {err}")
    except Exception as e:
        print(f"ERROR (chatbot_server): Unexpected error in get_item_info_by_identifier: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching item data: {str(e)}")


@app.get("/outlets_list")
def get_outlets_list(db: Session = Depends(get_db)) -> list[str]:
    try:
        outlets = db.query(Outlet.outlet_identifier).distinct().order_by(Outlet.outlet_identifier).all()
        results = [o[0] for o in outlets]
        print(f"DEBUG (chatbot_server): Fetched {len(results)} distinct outlet identifiers.")
        return results
    except SQLAlchemyError as err:
        print(f"ERROR (chatbot_server): SQLAlchemy error in get_outlets_list: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching outlet list: {err}")
    except Exception as e:
        print(f"ERROR (chatbot_server): Unexpected error in get_outlets_list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching outlet list: {str(e)}")


@app.get("/items_list")
def get_items_list(db: Session = Depends(get_db)) -> list[str]:
    try:
        items = db.query(Item.item_identifier).distinct().order_by(Item.item_identifier).all()
        results = [item[0] for item in items]
        print(f"DEBUG (chatbot_server): Fetched {len(results)} distinct item identifiers.")
        return results
    except SQLAlchemyError as err:
        print(f"ERROR (chatbot_server): SQLAlchemy error in get_items_list: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching item list: {err}")
    except Exception as e:
        print(f"ERROR (chatbot_server): Unexpected error in get_items_list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching item list: {str(e)}")


@app.get("/outlet_performance/{outlet_identifier}")
def get_outlet_performance_by_identifier(outlet_identifier: str, db: Session = Depends(get_db)) -> dict:
    try:
        print(f"DEBUG (chatbot_server): get_outlet_performance: Attempting to query for outlet_identifier: {outlet_identifier}")
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
        print(f"DEBUG (chatbot_server): get_outlet_performance: DB result for {outlet_identifier}: {outlet_data}")
        return outlet_data
    except SQLAlchemyError as err:
        print(f"ERROR (chatbot_server): SQLAlchemy error in get_outlet_performance_by_identifier: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching outlet performance: {err}")
    except Exception as e:
        print(f"ERROR (chatbot_server): Unexpected error in get_outlet_performance_by_identifier: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching outlet data: {str(e)}")


@app.get("/prediction_info/{item_identifier}/{outlet_identifier}")
def get_prediction_info(item_identifier: str, outlet_identifier: str, db: Session = Depends(get_db)) -> dict:
    try:
        print(f"DEBUG (chatbot_server): get_prediction_info: Attempting to query for item: {item_identifier}, outlet: {outlet_identifier}")
        prediction = (
            db.query(PerformancePrediction)
              .join(Item, Item.item_id == PerformancePrediction.item_id)
              .join(Outlet, Outlet.outlet_id == PerformancePrediction.outlet_id)
              .filter(Item.item_identifier == item_identifier,
                      Outlet.outlet_identifier == outlet_identifier)
              .order_by(PerformancePrediction.prediction_id.desc())  # latest by PK
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
        print(f"DEBUG (chatbot_server): get_prediction_info: DB result for {item_identifier}-{outlet_identifier}: {prediction_data}")
        return prediction_data
    except SQLAlchemyError as err:
        print(f"ERROR (chatbot_server): SQLAlchemy error in get_prediction_info: {err}")
        raise HTTPException(status_code=500, detail=f"Database error fetching prediction info: {err}")
    except Exception as e:
        print(f"ERROR (chatbot_server): Unexpected error in get_prediction_info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching prediction data: {str(e)}")


# PredictionInput: model DOES NOT require item_avg_sales or item_outlet_sales
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

    # Observation (NOT sent to model — storage/analytics only)
    item_outlet_sales: float = Field(..., description="Observed sales for this item at this outlet")

    # Optional helper (not sent to model)
    item_sales_deviation: float | None = Field(
        default=None,
        description="If provided, used directly; otherwise computed as item_outlet_sales - historical average"
    )
    always_add_new_rows: bool = Field(
        default=True,
        description="When True, inserts new rows into items/outlets/performance_predictions (append-only)"
    )


@app.post("/predict_item_sales")
async def predict_item_sales(prediction_input: PredictionInput, db: Session = Depends(get_db)):
    print(f"DEBUG (chatbot_server): /predict_item_sales input: {prediction_input.dict()}")

    if not DEPLOYED_ML_MODEL_URL:
        raise HTTPException(status_code=500, detail="Deployed ML Model URL is not configured.")

    # Compute historical average & deviation (for storage only)
    observed_sales = float(prediction_input.item_outlet_sales)

    # Find all historical item rows for this identifier (append-only means many items can exist)
    existing_item_ids = db.query(Item.item_id)\
                          .filter(Item.item_identifier == prediction_input.Item_Identifier)\
                          .all()
    existing_item_ids = [row[0] for row in existing_item_ids]

    past_avg = None
    if existing_item_ids:
        # Average of historical observed sales for this item:
        # Join Outlet (per-observation sales) with PerformancePrediction to link to item_ids
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

    # Call ML model (NO avg/sales fields included)
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
    print(f"DEBUG: ML payload (no avg/sales): {ml_payload}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(DEPLOYED_ML_MODEL_URL, json=ml_payload, timeout=60.0)
            response.raise_for_status()
        prediction_result = response.json()
        print(f"DEBUG (chatbot_server): Raw Prediction from ML Model: {prediction_result}")
    except httpx.RequestError as exc:
        print(f"ERROR (chatbot_server): ML request error: {exc}")
        raise HTTPException(status_code=503, detail=f"Could not connect to deployed ML model: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"ERROR (chatbot_server): ML bad status {exc.response.status_code}: {exc.response.text}")
        raise HTTPException(status_code=exc.response.status_code, detail=f"ML Model API error: {exc.response.text}")

    predicted_string_label = prediction_result.get("predicted_outlet_performance", "Unknown")
    predicted_label_int = PREDICTION_LABEL_MAPPING.get(predicted_string_label, 99)

    # Optional: some deployments also return predicted_sales_value
    predicted_sales_value = prediction_result.get("predicted_sales_value", None)
    try:
        predicted_sales_value = float(predicted_sales_value) if predicted_sales_value is not None else None
    except ValueError:
        predicted_sales_value = None

    # APPEND-ONLY inserts to all three tables
    try:
        # Always insert a new Item row
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
        print(f"DEBUG (chatbot_server): Inserted new item row id={item_id}")

        # Always insert a new Outlet row
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
        print(f"DEBUG (chatbot_server): Inserted new outlet row id={outlet_id}")

        # Always insert a new PerformancePrediction row
        new_prediction = PerformancePrediction(
            item_id=item_id,
            outlet_id=outlet_id,
            # Store the actual deviation (not the ML predicted sales)
            item_sales_deviation=computed_item_sales_deviation,
            performance_label=predicted_label_int,
            predicted_label=predicted_label_int
        )
        db.add(new_prediction)

        db.commit()
        db.refresh(new_prediction)
        print(f"DEBUG (chatbot_server): Inserted performance_predictions id={new_prediction.prediction_id}")

        storage_status = "Appended new rows to items, outlets, and performance_predictions."

    except SQLAlchemyError as err:
        print(f"ERROR (chatbot_server): SQLAlchemy storage error: {err}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database storage failed: {err}")
    except Exception as e:
        print(f"ERROR (chatbot_server): Unexpected error during storage: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Unexpected storage error: {e}")

    # Respond with details useful for the UI
    return {
        "prediction_label": predicted_string_label,
        "predicted_sales_value": predicted_sales_value,
        "observed_item_outlet_sales": observed_sales,
        "computed_item_avg_sales": computed_item_avg_sales,
        "computed_item_sales_deviation": computed_item_sales_deviation,
        "storage_status": storage_status
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
