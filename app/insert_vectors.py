from datetime import datetime

import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv("../data/all-outputs-fixed.csv", sep=",")


# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.

    This function creates a record with a UUID version 1 as the ID, which captures
    the current time or a specified time.

    Note:
        - By default, this function uses the current time for the UUID.
        - To use a specific time:
          1. Import the datetime module.
          2. Create a datetime object for your desired time.
          3. Use uuid_from_time(your_datetime) instead of uuid_from_time(datetime.now()).

        Example:
            from datetime import datetime
            specific_time = datetime(2023, 1, 1, 12, 0, 0)
            id = str(uuid_from_time(specific_time))

        This is useful when your content already has an associated datetime.
    """
    product_description = row["PRODUCT_DESCRIPTION"]
    embedding = vec.get_embedding(product_description)
    
    # Helper function to handle NaN values
    def clean_value(value):
        return None if pd.isna(value) else value
    
    return pd.Series({
        "id": str(uuid_from_time(datetime.now())),
        "metadata": {
            "product_id": row["PRODUCT_ID"],
            "product_name": row["PRODUCT_NAME"],
            "category": clean_value(row["PRODUCT_CATEGORY"]),
            "entity": clean_value(row["ENTITY"]),
            "card_type": clean_value(row["CARD_TYPE"]),  # Handle NaN
            "is_liability": clean_value(row["IS_LIABILITY"]),
            "is_investment": clean_value(row["IS_INVESTMENT"]),
            "user_commission": clean_value(row["USER_COMMISSION"]),
            "company_commission": clean_value(row["COMPANY_COMMISSION"]),
            "product_highlights": clean_value(row["PRODUCT_HIGHLIGHTS_JSON"]),
            "tags": clean_value(row["TAGS"]),
            "created_at": datetime.now().isoformat(),
        },
        "content": product_description,
        "embedding": embedding,
    })


records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)
