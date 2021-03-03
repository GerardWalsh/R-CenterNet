# frame: 15915 - VID_20210119_123453
# frame: 20000 - 20210119_121949, 25000
import cv2
video_filepath = '/home/gexegetic/Downloads/20210119_121949.mp4'
cap = cv2.VideoCapture(video_filepath)

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
delay = 1 # 1ms

for frame_number in range(0, int(frames)):
    print('visualising frame {}'.format(frame_number))
    flag, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    w, h, _ = frame.shape
    # print(frame.shape)
    # print(w, h)
    frame = cv2.resize(frame, (int(h/2), int(w/2)))
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(delay)
    if key == ord('s'):
        delay = 30
    elif key == ord('f'):
        delay = 5
    elif key == ord('p'):
        cv2.waitKey()
    elif key == ord('q'):
        print('Releasing {} and quitting . . . . . . . '.format(video_filepath))
        cap.release()
        break
    else:
        pass



