import os
import argparse
from sys import platform
from glob import glob
import csv

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

# DATASET_PATH = '/media/ml/Backup/ROB535_Perception/rob535-fall-2019-task-1-image-classification/images/trainval/0a4dd0b3-e034-48c1-aab5-023e22d6a73c'
DATASET_PATH = '/media/ml/Backup/ROB535_Perception/rob535-fall-2019-task-1-image-classification/images/test'

def image_id(iamge_abs_path):
    guid = iamge_abs_path.split('/')[-2]
    idx = iamge_abs_path.split('/')[-1].replace('_image.jpg', '')
    return '{}/{}'.format(guid, idx)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def detect(source, save_txt=False, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, weights, half = opt.output, opt.weights, opt.half
    label_list=[]
    bbox_list=[]

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, _ in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                best_conf, best_label, bestbbox = 0, 0, [0, 0, 0, 0]
                for *xyxy, conf, _, cls in det:
                    if conf > best_conf:
                        best_conf = conf
                        best_label = int(cls) + 1
                        bestbbox = '{},{},{},{}'.format(
                            int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                # use most confident label as label
                label_list.append('{},{}'.format(image_id(p), best_label))
                bbox_list.append('{},{}'.format(image_id(p), bestbbox))
            else:
                label_list.append('{},{}'.format(image_id(p), 0))
                bbox_list.append('{},{}'.format(image_id(p), '0,0,0,0'))

            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)

    print('Done. (%.3fs)' % (time.time() - t0))
    return label_list, bbox_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-sdc.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data_sdc/sdc.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='../darknet/backup/yolov3-sdc_final.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default=DATASET_PATH, help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--label-file', type=str, default='labels.csv', help='labels file name')
    parser.add_argument('--bbox-file', type=str, default='bbox.csv', help='bounding box file name') 
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    # parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    full_label_list = []
    full_bbox_list = []
    with torch.no_grad():
        guids = get_immediate_subdirectories(opt.source)
        for guid in guids:
            source = os.path.join(opt.source, guid)
            label_list, bbox_list = detect(source)
            full_label_list.extend(label_list)
            full_bbox_list.extend(bbox_list)
            print(len(full_label_list))

    with open(os.path.join(DATASET_PATH, opt.label_file), 'w') as f:
        f.writelines('guid/image,label\n')

        for row in full_label_list:
            f.writelines('{}\n'.format(row))

    with open(os.path.join(DATASET_PATH, opt.bbox_file), 'w') as f:
        f.writelines('guid/image,bbox_x,bbox_y,bbox_x,bbox_y\n')

        for row in full_bbox_list:
            f.writelines('{}\n'.format(row))