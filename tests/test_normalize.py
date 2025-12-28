from src.pipeline.normalize import normalize_plate_id


def test_normalize_plate_id_arabic_indic_digits() -> None:
    text = " ١٢٣٤٥٦٧٨٩٠ ABC "
    assert normalize_plate_id(text) == "1234567890ABC"
