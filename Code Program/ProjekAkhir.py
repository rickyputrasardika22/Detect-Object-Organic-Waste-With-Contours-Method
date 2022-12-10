import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tkinter import *
from tkinter import filedialog
import PIL.Image
import PIL.ImageTk
import cv2
import requests
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB4

input_size = (96,96)
#define input shape
channel = (3,)
input_shape = input_size + channel

def preprocess(img,input_size):
    
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

def buka():
    
    counter = 0
    path = filedialog.askopenfilename(filetypes=[("Image File",".jpeg")])
    return path

#Fungsi Browser Gambar
def browser():
    
    global img1, img2, lbl
    
    deteksi = buka()
    img1 = PIL.Image.open(deteksi)
    img1 = img1.resize((320, 240), PIL.Image.ANTIALIAS)
    img2 = PIL.ImageTk.PhotoImage(img1)
    
    lbl = Label(frame2, width=320, height=240, bg='white', image=img2)
    lbl.image = img2
    lbl.pack()
    
    b1['state'] = DISABLED
    b4['state'] = NORMAL
    b3['state'] = NORMAL
    
def delete():
    lbl.destroy()
    text1.config(text='')
    b1['state'] = NORMAL
    b4['state'] = DISABLED
    b3['state'] = DISABLED
    
#Fungsi Open Camera
def ambil():
    
    def nothing(x):
       # any operation
       pass
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
    cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-V", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
    cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

    font = cv2.FONT_HERSHEY_COMPLEX

    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L-H", "Trackbars")
        l_s = cv2.getTrackbarPos("L-S", "Trackbars")
        l_v = cv2.getTrackbarPos("L-V", "Trackbars")
        u_h = cv2.getTrackbarPos("U-H", "Trackbars")
        u_s = cv2.getTrackbarPos("U-S", "Trackbars")
        u_v = cv2.getTrackbarPos("U-V", "Trackbars")

        lower_red = np.array([l_h, l_s, l_v])
        upper_red = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)

        # Contours detection
        if int(cv2.__version__[0]) > 3:
            # Opencv 4.x.x
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Opencv 3.x.x
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            if area > 400:
                cv2.drawContours(frame, [approx], 0, (51, 248, 28), 5)

                if len(approx) == 3:
                    cv2.putText(frame, "Kulit Pisang", (x, y), font, 1, (51, 248, 28))
                elif len(approx) == 4:
                    cv2.putText(frame, "Ranting", (x, y), font, 1, (51, 248, 28))
                elif len(approx) == 8:
                    cv2.putText(frame, "Cangkang Telur", (x, y), font, 1, (51, 248, 28))
                elif len(approx) == 9:
                    cv2.putText(frame, "Potongan Wortel", (x, y), font, 1, (51, 248, 28))
                elif len(approx) == 11:
                    cv2.putText(frame, "Potongan Wortel", (x, y), font, 1, (51, 248, 28))                
                elif len(approx) == 5:
                    cv2.putText(frame, "Daun", (x, y), font, 1, (51, 248, 28))
                elif len(approx) == 10:
                    cv2.putText(frame, "Kulit Pisang", (x, y), font, 1, (51, 248, 28))                
                elif len(approx) == 7:
                    cv2.putText(frame, "Kulit Pisang", (x, y), font, 1, (51, 248, 28))
                
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
    
        img_counter=0
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        #while(1):
        # _,frame = cap.read()
        #cv2.imshow('Ambil',frame)
        k = cv2.waitKey(5) & 0xFF
        if k ==27:
            cap.release()
            cv2.destroyAllWindows()
            break
        elif k%256 ==32:

            img_nama = "/home/pi/Downloads/Object_ricky/Hasil{}.jpeg".format(img_counter)
            image2 = cv2.imwrite(img_nama, frame)
            print("{} written!".format(img_nama))
            img_counter+=1
            cv2.imshow(img_nama, frame)


#Fungsi Pengklasifikasian
def proses():
    
    model = load_model('/home/pi/Pengklasifikasian/Model/Model_SampahOrganik2.h5')
    
    im = img1
    X = preprocess(im,input_size)
    X = reshape([X])
    y = model.predict(X)
    
    labels = ['Cangkang Telur', 'Daun', 'Potongan Wortel', 'Kulit Pisang', 'Ranting']
    
    a = labels[np.argmax(y)]

    print(y)
    print('Akurasi:', np.max(y))
    text1.config(text=''+a)
    
#Fungsi Exit
def close():
    root.destroy()
    
        
root = Tk()
root.title('Sampah Organik')

if __name__ == '_main_':
    w = 400
    h = 500
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)

    root.geometry('%dx%d+%d+%d' % (w, h, x, y))

#Frame 1
frame1 = LabelFrame(root, text="Gambar", width=400, height=400)
frame1.pack(side = TOP)

frame2 = LabelFrame(frame1, width=320, height=240, bg="white")
frame2.pack(pady=20)

text1 = Label(frame1, width = 20, height = 1, bg = 'white')
text1.pack(side=TOP, pady =5, expand= YES)

b1 = Button(frame1, text="Browser", command=browser )
b1.pack(side=LEFT, pady=20, expand=YES)

b3 = Button(frame1, text='Delete', command=delete)
b3.pack(side=LEFT, pady=20, expand=YES )
b3['state'] = DISABLED

b2 = Button(frame1, text="Ambil", command=ambil)
b2.pack(side=LEFT, pady=20,expand=YES )

#Frame 2
frame5 = LabelFrame(root, text="Button")
frame5.pack(side = TOP)

b4 = Button(frame5, text="Proses", command=proses)
b4.pack(side=TOP, pady=5, padx=65)
b4['state'] = DISABLED

b5 = Button(frame5, text="Exit", command=close)
b5.pack(side=BOTTOM, padx=65)


root.mainloop()