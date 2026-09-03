def analyze_dataset(df):

    column_info = {}

    for column in df.columns:

        column_info[column] = {
            "dtype": str(df[column].dtype),
            "missing_values": int(df[column].isnull().sum()),
            "unique_values": int(df[column].nunique())
        }

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_info": column_info
    }