def populate_database():
    import os
    import shutil
    import sqlite3
    import pandas as pd
    import requests

    db_url = (
        "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    )
    local_file = "travel2.sqlite"
    backup_file = "travel2.backup.sqlite"
    overwrite = False
    if overwrite or not os.path.exists(local_file):
        response = requests.get(db_url)
        response.raise_for_status()
        with open(local_file, "wb") as f:
            f.write(response.content)
        shutil.copy(local_file, backup_file)
    conn = sqlite3.connect(local_file)
    # cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    print("example_time", example_time)
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    print("current_time", current_time)
    time_diff = current_time - example_time
    print("time_diff", time_diff)

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    del df
    del tdf
    conn.commit()
    conn.close()
