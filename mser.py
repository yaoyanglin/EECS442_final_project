import numpy as np
import cv2
import math
import os
import shutil
import json

def bboxToRectangle(bbox, horizontal, vertical):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = x1+bbox[2]
    y2=y1+bbox[3]
    x1-=horizontal
    x2+=horizontal
    y1-=vertical
    y2+=vertical
    return [max(math.floor(x1),0), max(0,math.floor(y1)), math.ceil(x2), math.ceil(y2)]

def recsOverlapVertically(rec1, rec2):
    return rec1[1]<=rec2[3] and rec2[1] <= rec1[3]

def recsOverlapHorizontally(rec1, rec2):
    return rec1[0]<=rec2[2] and rec2[0] <= rec1[2]

def mergeRecs(recs):
    grouped=[]
    valid_recs=[]
    for rec in recs:
        if rec[0] < rec[2] and rec[1] < rec[3]:
            valid_recs.append(rec)
    for rect in valid_recs:
        merge = 0;
        for each_part in grouped:
            if recsOverlapHorizontally(each_part, rect) and recsOverlapVertically(each_part, rect):
                merge = 1
                each_part[0] = min(rect[0], each_part[0])
                each_part[1] = min(rect[1], each_part[1])
                each_part[2] = max(rect[2], each_part[2])
                each_part[3] = max(rect[3], each_part[3])
        if (merge == 0):
            grouped.append(rect);
    return grouped

def getUniqueBoxes(bboxes,errorWidth,errorHeight):
    grouped=[]
    for box in bboxes:
        merge = 0;
        for each_part in grouped:
            if outside(box, each_part, errorWidth, errorHeight):
                merge = 1
            elif outside(each_part, box, errorWidth, errorHeight):
                merge = 1
        if (merge == 0):
            grouped.append(box);
    return grouped

def near(rec1, rec2, errorWidth, errorHeight):
    center1 = ((rec1[0]+rec1[2])*0.5,(rec1[1]+rec1[3])*0.5)
    center2 = ((rec2[0]+rec2[2])*0.5,(rec2[1]+rec2[3])*0.5)
    return (abs(center1[0] - center2[0]) < errorWidth and abs(center1[1] - center2[1]) < errorHeight)

def inside(point, rec):
    return rec[0] <=point[0] and point[0] <= rec[2] and rec[1] <= point[1] and point[1]<= rec[3]

def outside(bbox1, bbox2, horizontal, vertical):
    # true if bbox1 outside bbox2
    w=(bbox1[2] + bbox2[2])/2
    h=(bbox1[3]+bbox2[3])/2
    rec1=bboxToRectangle(bbox1, w*horizontal, h*vertical)
    rec2=bboxToRectangle(bbox2, -w*horizontal, -h*vertical)
    if rec1[0]<=rec2[0] and rec1[2] >=rec2[2] and rec1[3] >= rec2[3] and rec1[1] <=rec2[1]:
#        print(rec1)
#        print(rec2)
#        print('\n\n')
        return True
    return False

##############################################################

def cropping(filename, count):
    dirName = "Sample"+str(count)
    if not os.path.exists(dirName):
        print('created',filename)
    else:
        shutil.rmtree(dirName)
    os.mkdir(dirName)
    img=cv2.imread(filename)
    height, width, channels = img.shape

    blue=img[:,:,0]
    green=img[:,:,1]
    red = img[:,:,2]
    m=max(height, width)

    #pre process, denoising
    k = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, k, iterations=1)
    img = cv2.erode(img, k, iterations=1)

###########################################################
    #Ibg = np.ones((height, width), np.uint8)
    ## try color enhancement
    #for row in range(height):
    #    for col in range(width):
    #
    #        r=int(img[row,col,2])
    #        g=int(img[row,col,1])
    #        b=int(img[row,col,0])
    #        #Ibg[row,col]=255*max(g/float(r+g+b),b/float(r+g+b))
    #        Ibg[row,col]=255*max(0, min(g-r,g-b)/float(r+g+b),min(b-r,b-g)/float(r+g+b))
    #        """
    #        # Chromatic filter
    #        d=20
    #        f=(abs(r-g)+abs(g-b)+abs(b-r))/3*d
    #        if f < 1:
    #            # white pixel
    #            Ibg[row,col]=255
    #            """
###########################################################################################

    #top btm left right
    padded_img= cv2.copyMakeBorder(img,(m-height)//2,m-height-(m-height)//2,(m-width)//2,m-width-(m-width)//2,cv2.BORDER_CONSTANT,value=120)
    resized_img = cv2.resize(padded_img, (1000, 1000))
    # img[row, column, channel]

    #padded_gray= cv2.copyMakeBorder(Ibg,(m-height)//2,m-height-(m-height)//2,(m-width)//2,m-width-(m-width)//2,cv2.BORDER_CONSTANT,value=120)
    #resized_gray = cv2.resize(padded_gray, (700, 700))

    resized_gray= cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    vis = resized_img.copy()

    """
    edge = cv2.Canny(vis, 200, 500)
    vis[edge!=0]=0
    #now vis is drawn with edges
    """

    mser = cv2.MSER_create()
    #mser.setMinArea(100)
    #mser.setMaxArea(5000)
    coordinates, bboxes=mser.detectRegions(resized_gray)
    # coordinates = [[connected component coordinates1],[2],...]
#    bboxes = set()
#    for each_box in bboxes_list:
#        bboxes.add(tuple(each_box))
#    bboxes=list(bboxes)
    # now bboxes is a list of bbox(a tuple each)

    #cope with overlapping bboxes
    #sum_height = 0
    #count=0

#######################################################
#
    unique_tolerance = 0.1
    k=0
    for i in range(len(bboxes)):
        box1=bboxes[i]
        if box1[0]==-1:
            continue
    #    sum_height +=box1[3]
    #    count+=1
        for j in range(i+1, len(bboxes)):
            box2=bboxes[j]
            if box2[0]==-1 or box1[0]==-1:
                continue
            if outside(box1, box2, unique_tolerance , unique_tolerance):
                k+=1
                box2[0]=-1
            elif outside(box2, box1, unique_tolerance, unique_tolerance):
                k+=1
                box1[0]=-1
    old_bboxes = bboxes
    bboxes=[]
    print(len(old_bboxes))
    for box in old_bboxes:
        if box[0] != -1:
            w=float(box[2])
            h=float(box[3])
            if w >=5 and h >= 10 and w/h <=1.3 and h/w<=10:
                bboxes.append(box)
    print(k)
    print(len(bboxes))
    #bboxes=getUniqueBoxes(bboxes,0.01*width,0.01*height)
###########################################################

    errorrate = 0.1
    recs = []
    centers=[]
    i=0
    for box in bboxes:
        rec=(bboxToRectangle(box,errorrate*width, -errorrate*height))
        center = ((rec[0]+rec[2])*0.5,(rec[1]+rec[3])*0.5)
        recs.append(rec)
        centers.append((center,i))
        i+=1

###################################################################
    


    # cv2.fitline(centers, CV_DIST_L2, 0, 0.01, 0.01)
    #group 
    grouped = mergeRecs(recs)
    while (grouped != mergeRecs(grouped)):
        grouped = mergeRecs(grouped)
        print(len(grouped))
    print('finished')
    
    word_dict = dict()
    #{ith grouped part: [rec0, rec5,...],...}
    # order the recs
    
    for i in range(len(grouped)):
        this_word = []
        for j in range(len(centers)):
            if inside(centers[j][0], grouped[i]):
                this_word.append(j)
        word_dict[i]=this_word
    
#### try to combine i into a single letter
#    word_dict_keys = word_dict.keys()
#    for i in word_dict_keys:
#        letters = word_dict[i]
#        
#        for l1 in letters:
#            if l1 == 'i' or l1 == '.':
#                continue
#            for l2 in letters:
#                if l2 == 'i' or l2=='.'or l1=='.'or l1 == 'i':
#                    continue
#                # l1 and l2 are indexes of recs/centers
#                w1=recs[l1][2]-recs[l1][0]
#                w2=recs[l2][2]-recs[l2][0]
#                h1=recs[l1][3]-recs[l1][1]
#                h2=recs[l2][3]-recs[l2][1]
#                if abs(centers[l1][0]-centers[l2][0])<(w1+w2)*0.1:
#                    if h1/h2>=3:
#                        word_dict[i][l2]='.'
#                        word_dict[i][l1]='i'
#                    elif h2/h1>=3:
#                        word_dict[i][l1]='.'
#                        word_dict[i][l2]='i'                        
    print(word_dict)
    print("sorting....\n")
    # sort each letter in word dict based on their position, left to right
    word_dict_keys = word_dict.keys()
    for i in word_dict_keys:
        letters = word_dict[i]
        letters.sort(key=lambda rec: centers[rec][0][0])
    print(word_dict)
    with open(dirName+'/'+"dict.txt",'w') as file:
        file.write(json.dumps(word_dict))
    
#####################################################################    
    # draw the grouping boxes
    i = 0
    for rec in grouped:
        #y:y+h, x:x+w
#        cropped = vis[rec[1]:rec[3],rec[0]:rec[2]]
#        if not os.path.exists(filename+'_dir'):
#            print('created',filename)
#            os.mkdir(filename+'_dir')
#        cv2.imwrite(filename+'_dir/'+str(i)+'.jpg', cropped)
        # expand the bboxes, only horizontally, assuming horizontal labels
        cv2.rectangle(vis, (rec[0],rec[1]),(rec[2],rec[3]),(255,0,0),2)
        i+=1

    
################ coloring ############
# [TRY UNCOMMENT]
#    ## connected components
#    filtered_coordinates = []
#    for each_cord in coordinates:
#        bbox = cv2.boundingRect(each_cord)
#        x,y,w,h = bbox
##        if w< 5 or h < 10 or w/h > 1.3 or h/w > 10:
##            continue
#        filtered_coordinates.append(each_cord)
#    ## fill with colors
#    for each in filtered_coordinates:
#        x = each[:,0]
#        y = each[:,1]
#        vis[y, x] = (255,0,0)

    
    i = 0
    for bbox in bboxes:
        if bbox[0]:
            #if True:
            rec = bboxToRectangle(bbox, 0, 0)
            #y:y+h, x:x+w
            cropped = resized_img[rec[1]:rec[3],rec[0]:rec[2]]
            cv2.imwrite(dirName+'/'+str(i)+'.jpg', cropped)
            # expand the bboxes, only horizontally, assuming horizontal labels
            cv2.rectangle(vis, (rec[0],rec[1]),(rec[2],rec[3]),(0,0,255),2)
            
            i+=1

    #average_height = sum_height/count
    #num_lines, margin = divmod(height, average_height)
    #margin /=2



#    factor = 0
#    # draw the bboxes
#    i = 0
#    print(bboxes)
#    for bbox in bboxes:
#        if bbox[0] != False:
#            w=float(bbox[2])
#            h=float(bbox[3])
#            if w >=2 and h >=10 and w/h <=1.3 and h/w<=5:
#            #if True:
#                rec = bboxToRectangle(bbox, width*factor, 0)
#                #y:y+h, x:x+w
#                cropped = vis[rec[1]:rec[3],rec[0]:rec[2]]
#                if not os.path.exists(filename+'_dir'):
#                    print('created',filename)
#                    os.mkdir(filename+'_dir')
#                cv2.imwrite(filename+'_dir/'+str(i)+'.jpg', cropped)
#                # expand the bboxes, only horizontally, assuming horizontal labels
#                # cv2.rectangle(vis, rec[0],rec[1],(0,0,255),1)
#                i+=1
    


#[TRY UNCOMMENT] this is convex shapes
#    all=[]
#    for eachArray in coordinates:
#        eachArray.reshape(-1, 1, 2)
#        all.append(cv2.convexHull(eachArray))
#        #cv2.approxPolyDP(hulls[0], 0.01, closed)
#    cv2.polylines(vis, all, 1, (0, 0, 255)) 
#    # cv2.putText(vis, str('change'), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
#    # cv2.fillPoly(vis, all, (0, 255, 0))

    #cv2.imshow('edge',edge)
    # cv2.imwrite("test.png", vis)
    cv2.imwrite('multiple.jpg',vis)
    #cv2.imshow('img', vis)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    with open(dirName+'/'+"detection.txt",'w') as file:
       a="filename"+filename+"\n#Bboxes:"+str(len(bboxes))
       file.write(a)

def main():
    #put this file in any folder $ROOT, then create a folder $ROOT/test, 
    #and put all the test images inside ROOT/test, and run this .py file
    count=0
    for d in os.listdir(os.getcwd()+'/test_detection/'):
        try:
            cropping( 'test_detection/'+d, count)
            count+=1
        except:
            pass
#   cropping('test/31.png',37.1)

main()
     