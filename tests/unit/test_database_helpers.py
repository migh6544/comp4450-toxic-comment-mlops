from decimal import Decimal

from backend.app.database import to_dynamodb_types


def test_to_dynamodb_types_converts_nested_floats():
    converted = to_dynamodb_types({"latency": 12.5, "scores": {"toxic": 0.9}})
    assert converted == {
        "latency": Decimal("12.5"),
        "scores": {"toxic": Decimal("0.9")},
    }
