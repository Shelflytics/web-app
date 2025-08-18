import os
import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
load_dotenv()

DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_PORT = os.getenv("SUPABASE_DB_PORT")
ENCODED_DB_PASSWORD = quote_plus(DB_PASSWORD or "")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{ENCODED_DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ORM Models
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
    item_outlet_sales = Column(Float)
    item_sales_deviation = Column(Float)
    outlet_establishment_year = Column(Integer)

class PerformancePrediction(Base):
    __tablename__ = 'performance_predictions'
    prediction_id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey('items.item_id'), nullable=False)
    outlet_id = Column(Integer, ForeignKey('outlets.outlet_id'), nullable=False)
    item_sales_deviation = Column(Float)
    performance_label = Column(Integer)
    predicted_label = Column(Integer)

# Core helpers
def _ensure_tables():
    Base.metadata.create_all(bind=engine)

def _ensure_view():
    sql = """
    create or replace view v_predictions_flat_all as
    select
      pp.prediction_id,
      i.item_identifier,
      i.item_weight,
      i.item_fat_content,
      i.item_visibility,
      i.item_type,
      i.item_mrp,
      i.item_avg_sales,
      o.outlet_identifier,
      o.outlet_establishment_year,
      o.outlet_size,
      o.outlet_location_type,
      o.outlet_type,
      o.item_outlet_sales,
      o.item_sales_deviation,
      pp.performance_label,
      pp.predicted_label
    from performance_predictions pp
    join items   i on i.item_id   = pp.item_id
    join outlets o on o.outlet_id = pp.outlet_id;
    """
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)

def _all_three_tables_empty(db: Session) -> bool:
    items = db.query(Item).count()
    outlets = db.query(Outlet).count()
    preds = db.query(PerformancePrediction).count()
    return items == 0 and outlets == 0 and preds == 0

def _seed_from_csv(db: Session, csv_path: str):
    df = pd.read_csv(csv_path)

    # Items
    db.add_all([
        Item(
            item_identifier=row['Item_Identifier'],
            item_weight=row['Item_Weight'],
            item_fat_content=row['Item_Fat_Content'],
            item_visibility=row['Item_Visibility'],
            item_type=row['Item_Type'],
            item_mrp=row['Item_MRP'],
            item_avg_sales=row.get('item_avg_sales')
        )
        for _, row in df.iterrows()
    ])
    db.commit()

    # Outlets
    db.add_all([
        Outlet(
            outlet_identifier=row['Outlet_Identifier'],
            outlet_size=row['Outlet_Size'],
            outlet_location_type=row['Outlet_Location_Type'],
            outlet_type=row['Outlet_Type'],
            outlet_age=row['Outlet_Age'],
            item_outlet_sales=row['Item_Outlet_Sales'],
            item_sales_deviation=row['Item_Sales_Deviation'],
            outlet_establishment_year=row['Outlet_Establishment_Year']
        )
        for _, row in df.iterrows()
    ])
    db.commit()

    # Map identifiers to ids (latest wins if dupes exist — OK for initial seed)
    item_id_map = {i.item_identifier: i.item_id for i in db.query(Item).all()}
    outlet_id_map = {o.outlet_identifier: o.outlet_id for o in db.query(Outlet).all()}

    # Predictions
    preds = []
    for _, row in df.iterrows():
        iid = item_id_map.get(row['Item_Identifier'])
        oid = outlet_id_map.get(row['Outlet_Identifier'])
        if iid and oid:
            preds.append(PerformancePrediction(
                item_id=iid,
                outlet_id=oid,
                item_sales_deviation=row['Item_Sales_Deviation'],
                performance_label=row['Performance_Label'],
                predicted_label=row['Predicted_Label'],
            ))
    db.add_all(preds)
    db.commit()

def ensure_seed_and_view(csv_path: str):
    """
    Call this at app start. It will:
      1) create tables if missing
      2) seed once from CSV if ALL THREE tables are empty
      3) create/replace the joined VIEW
    """
    _ensure_tables()
    db = SessionLocal()
    try:
        if _all_three_tables_empty(db):
            _seed_from_csv(db, csv_path)
        _ensure_view()
    except SQLAlchemyError as e:
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    ensure_seed_and_view("final_dataset_with_numeric_labels.csv")