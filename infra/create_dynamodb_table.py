"""Create the DynamoDB predictions table for AWS or DynamoDB Local."""

from __future__ import annotations

import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    region = os.getenv("AWS_REGION", "us-east-1")
    table_name = os.getenv("DYNAMODB_TABLE", "comp4450-toxic-comment-predictions")
    endpoint_url = os.getenv("DYNAMODB_ENDPOINT_URL", "").strip() or None

    dynamodb = boto3.resource("dynamodb", region_name=region, endpoint_url=endpoint_url)
    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "prediction_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "prediction_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        print(f"Created table: {table_name}")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ResourceInUseException":
            print(f"Table already exists: {table_name}")
            return
        raise


if __name__ == "__main__":
    main()
