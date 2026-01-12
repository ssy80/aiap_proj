from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import pandas as pd

def get_db_conn(db) -> Engine:
    """
    Creates and returns a SQLite database engine.
    """
    engine = create_engine(f"sqlite:///{db}")
    return engine


def load_df_sql(engine: Engine, load_sql: str) -> pd.DataFrame:
    """
    Runs an SQL query and returns the result as a DataFrame.
    """
    df = pd.read_sql(load_sql, engine)
    return df
