"""DynamoDB persistence for predictions and feedback."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import boto3
from botocore.exceptions import ClientError



def to_dynamodb_types(value: Any) -> Any:
    """Recursively convert Python floats into DynamoDB-compatible Decimal values."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: to_dynamodb_types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_dynamodb_types(item) for item in value]
    return value


class PredictionRepository:
    def __init__(
        self,
        table_name: str,
        region_name: str,
        endpoint_url: str | None = None,
    ) -> None:
        resource = boto3.resource(
            "dynamodb",
            region_name=region_name,
            endpoint_url=endpoint_url,
        )
        self.table = resource.Table(table_name)
        self.table_name = table_name

    def save_prediction(self, record: dict[str, Any]) -> None:
        self.table.put_item(Item=to_dynamodb_types(record))

    def add_feedback(
        self,
        prediction_id: str,
        is_correct: bool,
        corrected_labels: list[str] | None,
        feedback_timestamp: str,
    ) -> None:
        values: dict[str, Any] = {
            ":received": True,
            ":correct": is_correct,
            ":timestamp": feedback_timestamp,
        }
        update = (
            "SET feedback_received = :received, is_correct = :correct, "
            "feedback_timestamp = :timestamp"
        )
        if corrected_labels is not None:
            update += ", corrected_labels = :labels"
            values[":labels"] = corrected_labels
        try:
            self.table.update_item(
                Key={"prediction_id": prediction_id},
                UpdateExpression=update,
                ExpressionAttributeValues=values,
                ConditionExpression="attribute_exists(prediction_id)",
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code == "ConditionalCheckFailedException":
                raise KeyError(prediction_id) from exc
            raise
