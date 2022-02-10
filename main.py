import numpy as np
import sys
import tracker
# from detector import Detector
import glob, cv2, torch

sys.path.insert(0, 'E:/2dTracking/CrowdDet-master/lib')
from utils import misc_utils, visual_utils

if __name__ == '__main__':

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # 初始化 yolov5
    # detector = Detector()

    # 打开视频
    # capture = cv2.VideoCapture('./video/test.mp4')
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')

    # 打开图片文件夹
    image_files = sorted(glob.glob('C:/Users/yinuowang3/Downloads/FOV02/20190416/19/*.jpg'))
    detecter_img_dir = 'E:/2dTracking/Waiting-time-estimation-for-passengers-in-subway-station/results/detecter_img/'
    id_img_dir = 'E:/2dTracking/Waiting-time-estimation-for-passengers-in-subway-station/results/CNet-0.99/'
    half_img_dir = 'E:/2dTracking/Waiting-time-estimation-for-passengers-in-subway-station/results/set_y1/'

    threshold = 0.99
    img_root = 'C:/Users/yinuowang3/Downloads/FOV02/20190416/19/'
    json_file = 'E:/2dTracking/CrowdDet-master/model/rcnn_fpn_baseline/outputs/eval_dump/dump-19-40.json'

    time_counting={}
    time_avg = []
    time_avg_idx = 0
    one_min_full = False
    ids=[89, 129, 111, 189, 305, 420, 425, 478, 491, 499, 541, 555, 642, 656, 781, 827, 867, 829, 1008]
    # im_tmp = None

    misc_utils.ensure_dir('outputs')
    records = misc_utils.load_json_lines(json_file)#[:args.number]

    for record in records:
        dtboxes = misc_utils.load_bboxes(
                record, key_name='dtboxes', key_box='box', key_score='score', key_tag='tag')
        gtboxes = misc_utils.load_bboxes(record, 'gtboxes', 'box')
        dtboxes = misc_utils.xywh_to_xyxy(dtboxes)
        gtboxes = misc_utils.xywh_to_xyxy(gtboxes)
        keep = dtboxes[:, -2] > threshold
        dtboxes = dtboxes[keep]

        img_path = img_root + record['ID'] + '.jpg'
        im = misc_utils.load_img(img_path)
        # print(img_path)
        # im=cv2.imread(img_path)
    # for f_idx in range(len(image_files)):
    #     # 读取每帧图片
    #     # _, im = capture.read()
    #     im=cv2.imread(image_files[f_idx])
    #     print('frame: ', f_idx)
        if im is None:
            break
        
        # 缩小尺寸，1080x720->960x540
        # im = cv2.resize(im, (960, 540))
        
        width = im.shape[1]
        height = im.shape[0]
        bboxes=[]
        for i in range(len(dtboxes)):
            one_box = dtboxes[i]
            one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                        min(one_box[2], width - 1), min(one_box[3], height - 1)])
            x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
            bbox=[x1,y1,x2,y2,'person',float(dtboxes[i][4])]
            bboxes.append(bbox)

        list_bboxs = []
        # bboxes = detector.detect(im)
        im_cpy = im
        
        # for bbox in bboxes:
        #     x1, y1, x2, y2, label, acc = bbox
        #     im_cpy = cv2.rectangle(im_cpy, (x1,y1), (x2,y2), (0,0,0), 5)
        time_frame = 0
        person_count = 0
        item_bbox_id=None

        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox
                # # check whether the person is the chosen one
                # if track_id in ids:
                #     # check whether the person is on the upper half of the image
                #     if (y1+y2)/2<=360 :
                #         item_bbox_id = item_bbox
                if track_id not in time_counting.keys():
                    time_counting[track_id] = 0
                else:
                    time_counting[track_id] += 5
                person_count+=1
                time_frame += time_counting[track_id]

            # draw the bboxes

            # # for id-checking
            # if item_bbox_id is not None:
            #     output_image_frame = tracker.draw_bboxes(im, [item_bbox_id], time_counting, line_thickness=None)
            # else:
            #     output_image_frame = im         
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, time_counting, line_thickness=None)
        # output_image_frame = im_cpy
        # cv2.imwrite(detecter_img_dir+str(f_idx)+'.jpg', output_image_frame)
        #     if f_idx > 0:
        #         output_image_frame = tracker.draw_bboxes(im_tmp, list_bboxs, time_counting, line_thickness=None)
        #     else:
        #         output_image_frame = im
        # else:
        #     if im_tmp is not None:
        #         output_image_frame = im_tmp
        #     else:
        #         output_image_frame = im
        else:
            output_image_frame = im
        cv2.imwrite(id_img_dir+str(record['ID'])+'.jpg', output_image_frame)
        
        if not one_min_full:
            time_avg.append(0)
        if time_frame == 0:
            time_avg[time_avg_idx] = 0
        else:
            time_avg[time_avg_idx] = float(time_frame) / person_count
        time_avg_idx = (time_avg_idx+1) % 12
        if time_avg_idx == 0:
            one_min_full = True

        print(float(sum(time_avg)) / len(time_avg))
        pass

        # 输出图片
        # output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            
            # 清空list
            list_bboxs.clear()

        #     pass
        # pass

        # text_draw = 'DOWN: ' + str(down_count) + \
        #             ' , UP: ' + str(up_count)
        # output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
        #                                  org=draw_text_postion,
        #                                  fontFace=font_draw_number,
        #                                  fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.imshow('demo', output_image_frame)
        cv2.waitKey(10)
        # im_tmp = im
    #     pass
    # pass

    # capture.release()
    cv2.destroyAllWindows()
