from supabase import create_client, Client
import csv
# Link:https://supabase.com/docs/reference/python/

SUPABASE_URL = 'https://yekrygkccujzdoirdlmv.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlla3J5Z2tjY3VqemRvaXJkbG12Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTM2MTg4NDcsImV4cCI6MjAwOTE5NDg0N30.ei-S5uT97lxsfqMcQWdSwpDx5rslrPyTRbjsYoNOj3Q'

url: str = SUPABASE_URL
key: str = SUPABASE_KEY
supabase: Client = create_client(url, key)

def parse_data():
    """
    Parse the data from the Supabase API
    """


    response = supabase.table('ToolScan').select("*").order('DateTime').execute()

    # Specify the CSV file name
    csv_file = 'output.csv'

    # List of field names
    fieldnames = ['DateTime', 'ToolScanTime', 'Zone']

    # Write the data to the CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        for row in response.data:
            writer.writerow(row)

    print(f"Data has been successfully parsed")


def update(HQ_pred, AZ_pred):
    data, count = supabase.table('RepairPrediction').upsert({'Zone': 'HQ', 'DateTime': str(HQ_pred)}).execute()
    data, count = supabase.table('RepairPrediction').upsert({'Zone': 'AZ', 'DateTime': str(AZ_pred)}).execute()
