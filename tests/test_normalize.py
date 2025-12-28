from src.pipeline.normalize import normalize_plate_id


def test_normalize_plate_id_arabic_indic_digits() -> None:
    text = " \u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669\u0660 ABC "
    assert normalize_plate_id(text) == "1234567890ABC"
