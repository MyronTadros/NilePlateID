import os
import pathlib


# clean up the OCR output by removing unwanted characters
def post_process(input_file_path):
    # read the ocr result file
    f = open(input_file_path, "r")
    text = f.readline()
    
    # list of all punctuation and special chars we want to remove
    # license plates shouldnt have these
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',
              '#', '*', '+', '\\', '•', '~', '@', '£',
              '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′',
              'Â', '█', '½', '…',
              '"', '★', '"', '–', '●', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
              '¥', '▓', '—', '‹', '─',
              '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', ''', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '¾',
              'Ã', '⋅', ''', '∞',
              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'Ø',
              '¹', '≤', '‡', '√', '«', ' ']

    # remove each punctuation from the text
    for char in puncts:
        if char in text:
            text = text.replace(char, '')

    # write cleaned text back to file
    f.close()
    f = open(input_file_path, "w")
    f.write(text)
    f.close()
    return text


# OCR function using easyocr library
# easyocr works well for arabic text
def recognise_easyocr(src_path, out_path, langs=['ar', 'en']):
    # try to import easyocr
    try:
        import easyocr
    except:
        print('need to install easyocr first')
        return -1

    try:
        # create reader with specified languages
        # gpu=True makes it faster if you have a GPU
        reader = easyocr.Reader(langs, gpu=True)
        
        # read text from image
        results = reader.readtext(src_path, detail=0)
        text = ' '.join(results)
        
        # save result to file
        out_file = out_path + '.txt'
        f = open(out_file, 'w', encoding='utf-8')
        f.write(text)
        f.close()
        return 0
    except Exception as e:
        print('easyocr error:', e)
        return -1


# Tesseract OCR function for comparison with easyocr
def recognise_tesseract(src_path, out_path, lang='ara'):
    """
    Run tesseract OCR on image and save result to file.
    Using script-based Arabic model for better recognition.
    Returns 0 on success, -1 on failure.
    """
    # make sure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # find tesseract executable
    if os.name == 'nt':
        # windows
        tesseract_cmd = str(pathlib.Path().absolute() / 'extra' / 'tesseract.exe')
    else:
        # mac/linux
        tesseract_cmd = 'tesseract'
        from shutil import which
        if which('tesseract') is None:
            # try to find bundled version
            bundled = str(pathlib.Path().absolute() / 'extra' / 'tesseract.exe')
            if os.path.exists(bundled) and os.access(bundled, os.X_OK):
                tesseract_cmd = bundled
            else:
                print('Error: tesseract not found. Install with `brew install tesseract` or place an executable at extra/tesseract.exe')
                return -1

    # run tesseract command
    # psm 6 = assume a single uniform block of text
    tessdata_dir = os.environ.get("TESSDATA_PREFIX")
    cmd = [tesseract_cmd, src_path, out_path, '-l', lang, '--psm', '6', '--dpi', '300', '--oem', '1']
    if tessdata_dir:
        cmd.extend(['--tessdata-dir', tessdata_dir])

    try:
        import subprocess
        subprocess.run(cmd, check=True)
    except Exception as e:
        print('Error running tesseract:', e)
        return -1

    return 0
