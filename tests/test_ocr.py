# test ocr functions
import numpy as np
from src.pipeline.ocr import post_process, read_plate_text


def test_post_process():
    # test text cleaning
    text = "ABC-123, test!"
    result = post_process(text)
    assert result == "ABC123test"
    print("post_process test passed:", result)


def test_post_process_special_chars():
    # test removing special characters
    text = "AB@C#123$%"
    result = post_process(text)
    assert result == "ABC123"
    print("post_process special chars test passed:", result)


def test_read_plate_text_empty():
    # test with empty image
    empty_img = np.array([])
    text, conf = read_plate_text(empty_img)
    assert text == ""
    assert conf == 0.0
    print("read_plate_text empty test passed")


if __name__ == "__main__":
    test_post_process()
    test_post_process_special_chars()
    test_read_plate_text_empty()
    print("\nAll tests passed!")
