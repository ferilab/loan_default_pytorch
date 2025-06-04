
from sklearn.preprocessing import LabelEncoder

def process_data(df):
    # 1 Impute missing target values

    df.ChgOffDate.fillna(-1, inplace = True)
    df['Was_Charged_Off'] = df.ChgOffDate.apply(lambda x: 0 if x == -1 else 1)

    # 2 Make other binary (Y and N) categorical variables consistent. We kn02 0 means N but A or S can't be interpreted.

    df.RevLineCr = df.RevLineCr.apply(lambda x: "N" if x == "0" else "Y" if x == "T" else x)
    # First convert values to string and remove possible leading or trailing spaces
    df.LowDoc = df.LowDoc.astype(str).str.strip()
    df.LowDoc = df.LowDoc.apply(lambda x: "N" if x == "0" else float('nan') if x in ["A", "S"] else x)

    # 3 Pruning

    # Drop irrelevant 
    df.drop(columns=["Name", "City", "Bank", "BankState", "NAICS", "ApprovalDate", 
                    "ApprovalFY", "Zip", "State"], inplace=True, errors='ignore')

    # Drop the few observations with missing column values
    df.dropna(inplace=True)

    # 4 Encoding

    # Encode target
    df["MIS_Status"] = LabelEncoder().fit_transform(df["MIS_Status"])

    # Encode categoricals
    for col in df.select_dtypes(include="object"):
        df[col] = LabelEncoder().fit_transform(df[col])

    return df