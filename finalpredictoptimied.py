import os
import numpy as np
import tflite_runtime.interpreter as tflite
# import tensorflow as tf
import cv2
from time import sleep
import time
import requests
import random
from pathlib import Path
from collections import OrderedDict,namedtuple

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="yolov7_model_500ch_60epoch.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



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
names = ["abpnews","aljazeera","asianetnews","bbcworld","harekrsna","jayamax","jayaplus","kairalinews","kalaingartv","ksirippoli","lokshahimarathi","wion","adithyatv","chintutv","chuttitv","cnnint","etnow","etnowswadesh","geminicomedy","geminilife","geminitv","khushitv","ktv","mirrornow","sunlife","sunmusic","suntv","suryacomedy","suryamusic","suryatv","udayacomedy","udayamusic","udayatv","vh1","&flix","amritatv","andpictures","arirang","bhaktisagar2","colorsgujcin","colorssuper","dagdushethgnesh","ddbharati","ddkisan","ddnational","ddnews","etvap","fashiontv","gyandarshan","natgeowild","nationalgeographic","sansadtvsd","songdew","zeeaction","zeeanmol","zeeanmolcinema","zeebangla","zeebanglacin","zeebollywood","zeecafe","zeecinema","zeeclaassic","zeeganga","zeekannada","zeemarathi","zeepunjabi","zeetalkies","zeetamil","zeetelugu","zeeyouva","zeezest","zing","9xjhakaas","9xm","aadinathtv","aajtak","aakashaath","aastha","aasthabhajan","animalplanet","arihanttv","asianet","asianetmovies","asianetplus","b4ukadak","b4umoviies","bigmagic","bindass","cartoonnetwork","cbeebies","channelwin","chardiklatimetv","citynews","cnnnews18","colkanada","colorsbangla","colorsgujarati","colorsinfinity","colorsmarathi","dabangg","dangal","ddbangla","ddbihar","ddkashir","ddoriya","ddrajasthan","ddretro","ddsahyadri","dduttarpradesh","denaakhyaan","denaction","denbhakti","denbhojpuri","denboxoffice","denchalchitra","dencinema","denclassic","dencomedy","dengeetmala","dengurjari","denmarathi","dhamaal","dhinchaak2","dibaadat","discovery","discoverykids","discoveryscience","discoveryturbo","disneychannel","disneyjunior","djay","djayretro","dsindhi","epic","etvbalbharat","etvtelugu","eurosport","faktmarathi","foxlife","godtv","goldminesbollwwood","greenchillies","gujaratlivedarshan","gujfirst24x7","historytv18","hometheater","hungama","id","indiatoday","jinvani","kairali","kartavya","kflickskorean","kflix","lakshyatv","lordbuddha","maiboli","mantavyanews","mastii","mh1shraddha","mmanorama","mnx","moviesnow","mtv","nepal1","news12guj","news18jkhl","news18lokmat","news18phbr","news18upuk","nick","nickjr","nt1","otv","pogo","pravaahpicture","ptcchakde","ptcpunjabi","rajdigplus","rajmusixkann","rajnewskanada","rajnewstel","rbharat","republictv","romedynow","saamtv","sandeshnews","sangeetbangla","shemaroomarathi","shemarootv","shemarooumang","shubhsandesh","sohamtv","sonic","sonten3","sonyaath","sonybbceath","sonyenttv","sonymarathi","sonymax","sonymax2","sonypal","sonypix","sonysab","sonyten1","sonyten2","sonyten5","sonywah","sonyyay","sports181","starbharat","stargold","stargold2","stargoldselect","starjalsa","starmaa","starmaagold","starmaamusic","starmovies","starplus","starpravah","starspo1hindi","starsports1","starsports2","starsports3","starsportsfirst","starsportsselect2","starsposel1","starsuvarna","starsuvplus","starutsav","starutsavmovies","starvijay","starworld","superhungama","svbc","tarangmusic","tehzeebtv","theq","timesnownavbhar","tlc","travelxp","tv9bharatvarsh","tv9gujarati","tv9kannada","tv9marathi","tv9telugu","utvaction","utvmovies","valamtv","vtvgujarati","zee24taas","zeenews","zeephhnews","zeesalaam","zeetv","zoom","abpananda","abpasmita","abpganga","abpmaza","abpsanjha","cna","goodnewstoday","isaiaruvi","janamtv","jmovies","manoramanews","mathrubhuminews","mega24","megamusic","megatv","murasutv","russiatoday","sethigaltv","siripollitv","australiatv","cnbcaawaa","cnbcbazaar","cnbctv18","cnn","ddchandana","ddgirnar","ddindia","ddmadhyapradesh","ddmalayalam","ddpodhigai","ddsaptgiri","ddyadgiri","france24","geminimovies","geminimusic","kochutv","kolkatatv","kushitv","malaimurasuseithigirl","mediaone","news7tamil","polimernews","polimertv","publictv","puthiyathalaimurai","rajnewstamil","sathyamtv","sunbangla","sunmarathi","sunnews","suryamovies","thanthitv","timesnow","udayamovies","vendhartv","colorsbanglacinema","colorskannadacinema","ddgyandarshan","dw","etvtelangana","kappatv","newsnation","newsstateup","safaritv","tv5kannada","tv5mondeasie","tv5news","wetv","zeebiscope","zeecinemalu","zeekeralam","zeepicchar","zeesarthak","10tv","1sports","1stindiarajasthan","24","4tv","9xjalwa","abnAndhraJyoti","alankar","anaaditv","anbnews","angeltv","anjantv","apnnews","aradanatv","argusnews","asianetsuvarnanews","awakening","ayushtv","b4ubhojpuri","b4umusic","bansalnews","bflix","bhaktitv","bharatsamachar","bhojpuricinema","calcuttanews","cinematvindia","colorscineplex","colorsoriya","colorsrishtey","colorstamil","comedycentral","ctvn","dangal2","ddarunprabha","ddmanipur","ddnortheast","ddpunjabi","ddsports","ddurdu","dighvijaynews","divya","dtamil","dy365","e24","enter10bangla","etvabhiruchi","etvcinema","etvlife","etvplus","ezmall","filamchibhojpuri","flowers","foodfood","goldmines","goodness","goodtimes","gstv","gubbare","gulistannews","harvesttv24x7","hindikhabar","hindudharam","hmtv","ibc24","indianews","indianewsgujarat","indiatv","indiavoice","indradhanu","ing24x7","ishara","ishwartv","jaihindtv","jaimaharashtra","jantatv","jantntratv","jantv","jeevantv","jothitv","kalingatv","kanaknews","kannadanaaptol","kashishnews","kasthuri","kaumudytv","khabreinabhitak","khushboobangla","livingindianews","madhatv","makkaltv","manoranjangrand","manoranjantv","mh1","mh1dilse","mh1news","mtvbeats","nambikkaltv","nandighoshatv","ndtv24x7","ndtvindia","ndtvprofitprime","network10","news18assamnortheast","news18bangla","news18biharjharkhand","news18gujarati","news18india","news18kannada","news18kerala","news18odia","news18rajasthan","news18tamilnadu","news1india","news24","news24mp","newsfirst","newsindia","newsj","newslive","newstimebangla","newsx","nktvplus","northeastlive","ntvtelugu","oscarmovies","parasgold","pasand","patrikatv","peaceofmind","pepperstv","pitaara","powertv","pragnews","prameyanews7","prarthanatv","pratidintime","primenews","ptcmusic","ptcnews","ptcpunjabigold","ptcsimran","publicmovies","publicmusic","punjabihits","rajmusix","rajmusixmalayalam","rajmusixtelugu","rajnewsmalayalam","rajtv","ramdhenu","rangtv","rengonitv","republicbangla","ruposhibangla","sadhnanewsmpcg","sadhnaplusnews","sadhnatv","sakshitv","samay","sangeetmarathi","sankaratv","sanskaar","santwani","satsang","shalomtv","sharnamtv","showbox","shubhtv","srikannadaalltime","starjalsamovies","starmaamovies","studioyuva","subhartitv","subhavaarthatv","sudarshannews","svbc2","swarajexpresssmbc","swarasagar","tarangtv","tnews","totaltv","travelxptamil","tv9bangla","v6telugu","vasanth","vedic","vijaysuper","vijaytakkar","vissatv","wowcinemaone","zee24ghanta","zee24kalaak","zeebiharjharkhand","zeebusiness","zeehindustan","zeemadhyapradeshchhatisgarh","zeerajasthan","zeeuttarpradeshuttarakhand"]


cap = cv2.VideoCapture(1)
cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

	

kd = 1
while True:
	start = time.time()
	
	ret,frame = cap.read()

	tl = frame[0:149,0:200]
	bl = frame[331:480,0:200]
	tr = frame[0:149,440:640]
	br = frame[331:480,440:640]
	v1 = cv2.vconcat([tl,bl])
	v2 = cv2.vconcat([tr,br])
	img = cv2.hconcat([v1,v2])	

	del frame
	del tl
	del tr
	del bl
	del br
	del v1
	del v2

	image, ratio, dwdh = letterbox(img, auto=False)

	del img

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = image.transpose((2, 0, 1))
	image = np.expand_dims(image, 0)
	image = np.ascontiguousarray(image)
	image = image.astype(np.float32)
	image /= 255


	interpreter.reset_all_variables()
	interpreter.set_tensor(input_details[0]['index'], image)
	interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
	output_data = interpreter.get_tensor(output_details[0]['index'])

	channel_name = names[int(output_data[0][5])]
	print(channel_name)
	channel_probab = output_data[0][6]
	print(channel_probab)    

	del image
	kd = kd + 1
