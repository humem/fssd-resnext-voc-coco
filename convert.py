#!/usr/bin/env python

from glob import glob
import os
import re

anno_dir = '../datasets/VOCdevkit/VOC2007/Annotations'
image_dir = '../datasets/VOCdevkit/VOC2007/JPEGImages'

anno_paths = glob(anno_dir + '/*.xml')
anno_paths.sort()
print(f'{len(anno_paths)} annotation')

regexp = re.compile(r'<name>person</name>.*<xmin>(\d+)</xmin>\s*<ymin>(\d+)</ymin>\s*<xmax>(\d+)</xmax>\s*<ymax>(\d+)</ymax>')

data = []
for path in anno_paths:
    with open(path) as f:
        text = f.read()
    text = text.replace('\n', ' ')
    m = re.search(r'\s*<filename>(.+)</filename>', text)
    image_path = os.path.join(image_dir, m.group(1))
    bboxes = []
    cursor_start = 0
    cursor_end = 0
    while True:
        cursor_start = text.find('<object>', cursor_end)
        if cursor_start < 0:
            break
        cursor_end = text.find('</object>', cursor_start)
        object_text = text[cursor_start:cursor_end]
        #print(cursor_start, cursor_end, object_text)
        m = regexp.search(object_text)
        #print(m)
        if m:
            bboxes.append(','.join([m.group(i) for i in range(1, 5)]))
    # xmin,ymax,xmax,ymax,label;xmin,...
    if len(bboxes) > 0:
        anno = ';'.join([bbox + ',0' for bbox in bboxes])
        data.append('\t'.join([image_path, anno]))
        #print(image_path)
        #print(bboxes)
        #print(anno)
        #break

dataset_path = 'data/det/data.tsv'
print(f'Save {len(data)} data in {dataset_path}')
with open(dataset_path, 'w') as f:
    f.write('\n'.join(data) + '\n')
