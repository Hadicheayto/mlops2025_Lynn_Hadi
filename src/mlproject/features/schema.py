# src/mlproject/features/schema.py

CATEGORICAL_COLS = [
    "vendor_id",
    "store_and_fwd_flag",
]

NUMERICAL_COLS = [
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "trip_distance_km",
    "manhattan_distance",
    "is_night",
]

COLUMNS_TO_DROP = [
    "id",
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]