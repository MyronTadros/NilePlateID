from src.pipeline.associate import match_plates_to_cars


def test_match_plates_to_cars_prefers_higher_iou() -> None:
    cars = [
        {"id": "car1", "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]},
        {"id": "car2", "bbox_xyxy": [30.0, 30.0, 70.0, 70.0]},
    ]
    plates = [
        {"id": "plate1", "bbox_xyxy": [40.0, 40.0, 60.0, 60.0]},
    ]

    matches = match_plates_to_cars(cars, plates)

    assert len(matches) == 1
    assert matches[0]["car"]["id"] == "car2"


def test_match_plates_to_cars_no_match_when_outside() -> None:
    cars = [{"id": "car1", "bbox_xyxy": [0.0, 0.0, 50.0, 50.0]}]
    plates = [{"id": "plate1", "bbox_xyxy": [60.0, 60.0, 80.0, 80.0]}]

    matches = match_plates_to_cars(cars, plates)

    assert matches == []
