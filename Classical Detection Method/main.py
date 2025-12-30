import cv2
import numpy
import argparse
import os
import glob
import sys
import time
from plate_detection import detect, process
from ocr_processing import recognise_easyocr, recognise_tesseract, post_process



parser = argparse.ArgumentParser()
parser.add_argument('--i', help="image file")
parser.add_argument('--v', help="video file")

args = parser.parse_args()


# process image files
if args.i:
    start_time = time.time()  # start timer
    
    # create temp folders if they dont exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    if not os.path.exists('temp/steps'):
        os.makedirs('temp/steps')

    # delete old files from previous runs
    files = glob.glob('temp/crop*.jpg')
    files = files + glob.glob('temp/crop*.txt')
    files = files + glob.glob('temp/steps/*')
    for f in files:
        try:
            os.remove(f)
        except:
            pass  # if file doesnt exist just skip it

    # read the image
    img = cv2.imread(args.i)
    
    # call detect function from plate_detection module
    result, crops = detect(img)

    # process each detected plate
    count = 1
    for crop in crops:
        # save original crop
        cv2.imwrite('temp/crop_original_' + str(count) + '.jpg', crop)
        
        # process it to make it better for ocr
        processed_crop = process(crop, count)
        cv2.imwrite('temp/crop' + str(count) + '.jpg', processed_crop)

        # do ocr with easyocr
        recognise_easyocr('temp/steps/plate' + str(count) + '_6_threshold.png', 'temp/crop'+str(count)+'_easyocr')
        
        # do ocr with tesseract for comparison
        recognise_tesseract('temp/steps/plate' + str(count) + '_6_threshold.png', 'temp/crop'+str(count)+'_tesseract')

        # clean up the easyocr output
        post_process('temp/crop' + str(count) + '_easyocr.txt')
        
        # clean up the tesseract output
        post_process('temp/crop' + str(count) + '_tesseract.txt')
        count = count + 1
        
    # save result image with detections drawn
    cv2.imwrite('temp/detection.jpg', result)
    
    # calculate time taken
    end_time = time.time()
    print('Processing time: '+ str(end_time - start_time) + ' seconds')
    
elif args.v:
    # process video files
    video = cv2.VideoCapture(args.v)

    # loop through all frames
    while True:
        ret, frame = video.read()
        if ret:
            # detect plates in frame
            frame, crop = detect(frame)
            
            # show instructions on screen
            cv2.putText(frame, 'Press Q to quit',(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255), 2)
            cv2.imshow('Video', frame)

            # quit if Q is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

else:
    print("use --i for image or --v for video")

	