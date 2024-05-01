import os
import boto3
from dotenv import load_dotenv

load_dotenv()

# Get the service resource.
dynamodb = boto3.resource(
    service_name='dynamodb',
    endpoint_url="http://127.0.0.1:8000/",
    aws_access_key_id=os.getenv('STATIC_KEY_ID'),
    aws_secret_access_key=os.getenv('STATIC_KEY_SECRET'),
    region_name='ru-central1',
)

# Create the DynamoDB table.
table = dynamodb.create_table(
    TableName="SessionTable",
    KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
    AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
    BillingMode="PAY_PER_REQUEST",
)

# Wait until the table exists.
table.meta.client.get_waiter("table_exists").wait(TableName="SessionTable")

# Print out some data about the table.
print(table.item_count)
