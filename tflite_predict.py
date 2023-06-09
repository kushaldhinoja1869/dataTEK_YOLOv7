import cv2
import time
import requests
import random
import numpy as np
# import onnxruntime as ort
# from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2 as cv
import os
import cv2
import numpy as np
import glob
import random



# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="yolov7_model_tiny_120epochs.tflite")





def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

#Name of the classes according to class indices.
names = ["abpnews","aljazeera","asianetnews","bbcworld","harekrsna","jayamax","jayaplus","kairalinews","kalaingartv","ksirippoli","lokshahimarathi","wion","adithyatv","chintutv","chuttitv","cnnint","etnow","etnowswadesh","geminicomedy","geminilife","geminitv","khushitv","ktv","mirrornow","sunlife","sunmusic","suntv","suryacomedy","suryamusic","suryatv","udayacomedy","udayamusic","udayatv","vh1","&flix","amritatv","andpictures","arirang","bhaktisagar2","colorsgujcin","colorssuper","dagdushethgnesh","ddbharati","ddkisan","ddnational","ddnews","etvap","fashiontv","gyandarshan","natgeowild","nationalgeographic","sansadtvsd","songdew","zeeaction","zeeanmol","zeeanmolcinema","zeebangla","zeebanglacin","zeebollywood","zeecafe","zeecinema","zeeclaassic","zeeganga","zeekannada","zeemarathi","zeepunjabi","zeetalkies","zeetamil","zeetelugu","zeeyouva","zeezest","zing","9xjhakaas","9xm","aadinathtv","aajtak","aakashaath","aastha","aasthabhajan","animalplanet","arihanttv","asianet","asianetmovies","asianetplus","b4ukadak","b4umoviies","bigmagic","bindass","cartoonnetwork","cbeebies","channelwin","chardiklatimetv","citynews","cnnnews18","colkanada","colorsbangla","colorsgujarati","colorsinfinity","colorsmarathi","dabangg","dangal","ddbangla","ddbihar","ddkashir","ddoriya","ddrajasthan","ddretro","ddsahyadri","dduttarpradesh","denaakhyaan","denaction","denbhakti","denbhojpuri","denboxoffice","denchalchitra","dencinema","denclassic","dencomedy","dengeetmala","dengurjari","denmarathi","dhamaal","dhinchaak2","dibaadat","discovery","discoverykids","discoveryscience","discoveryturbo","disneychannel","disneyjunior","djay","djayretro","dsindhi","epic","etvbalbharat","etvtelugu","eurosport","faktmarathi","foxlife","godtv","goldminesbollwwood","greenchillies","gujaratlivedarshan","gujfirst24x7","historytv18","hometheater","hungama","id","indiatoday","jinvani","kairali","kartavya","kflickskorean","kflix","lakshyatv","lordbuddha","maiboli","mantavyanews","mastii","mh1shraddha","mmanorama","mnx","moviesnow","mtv","nepal1","news12guj","news18jkhl","news18lokmat","news18phbr","news18upuk","nick","nickjr","nt1","otv","pogo","pravaahpicture","ptcchakde","ptcpunjabi","rajdigplus","rajmusixkann","rajnewskanada","rajnewstel","rbharat","republictv","romedynow","saamtv","sandeshnews","sangeetbangla","shemaroomarathi","shemarootv","shemarooumang","shubhsandesh","sohamtv","sonic","sonten3","sonyaath","sonybbceath","sonyenttv","sonymarathi","sonymax","sonymax2","sonypal","sonypix","sonysab","sonyten1","sonyten2","sonyten5","sonywah","sonyyay","sports181","starbharat","stargold","stargold2","stargoldselect","starjalsa","starmaa","starmaagold","starmaamusic","starmovies","starplus","starpravah","starspo1hindi","starsports1","starsports2","starsports3","starsportsfirst","starsportsselect2","starsposel1","starsuvarna","starsuvplus","starutsav","starutsavmovies","starvijay","starworld","superhungama","svbc","tarangmusic","tehzeebtv","theq","timesnownavbhar","tlc","travelxp","tv9bharatvarsh","tv9gujarati","tv9kannada","tv9marathi","tv9telugu","utvaction","utvmovies","valamtv","vtvgujarati","zee24taas","zeenews","zeephhnews","zeesalaam","zeetv","zoom"]

#Creating random colors for bounding box visualization.
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}




ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
mypath = ROOT_DIR + '\\' + "test"
storepath = ROOT_DIR + '\\' + "testresult"

for filePath in glob.glob(mypath + "\\*.jpg"):
    start = time.time()
    namekd = filePath.split("\\")[-1]
    #Load and preprocess the image.
    img = cv2.imread(filePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255


    #Allocate tensors.
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], im)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    print(time.time() - start)


    ori_images = [img.copy()]

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(output_data):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
    
    imf = cv2.cvtColor(ori_images[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(storepath +"\\"+namekd,imf)
    