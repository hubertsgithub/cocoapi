from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pylab

import PIL.Image
import PIL.ImageDraw
import tqdm

### Based on pycocoDemo.ipynb ###
dataDir='..'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())

# coco id to coco id
coco_id_to_name = {entry['id']: entry['name'] for entry in cats}
# coco name to pascal name (in case some names aren't exact match)
coco_name_to_pascal_name = {} # TODO
# pascal name to pascal id
pascal_name_to_id = {} # TODO

#nms=[cat['name'] for cat in cats]
#print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
#nms = set([cat['supercategory'] for cat in cats])
#print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
# Choose categories corresponding to pascal voc categories
catNms = ['person','dog','skateboard']
catIds = coco.getCatIds(catNms);

print 'Gathering all annotations with classes in...\n\t{}'.format(catNms)
annIds = coco.getAnnIds(catIds=catIds)
anns = coco.loadAnns(annIds)
print 'Getting img ids corresponding to annotations...'
img_ids = set([ann['image_id'] for ann in anns])
print '\t{} imgs.'.format(len(img_ids))

print 'Gathering images...'
imgs = coco.loadImgs(img_ids)

print 'Generating label pngs...'
for img in tqdm.tqdm(imgs[1000:1010]):

    plt.clf()
    # load and display instance annotations for a single image
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    all_masks = None
    for ann in anns:
        # This generates binary mask.
        mask = coco.annToMask(ann)
        #print set(mask.flatten())
        coco_class_id = ann['category_id']
        mask = mask.astype(np.uint8) * coco_class_id
        if all_masks is None:
            all_masks = mask
        else:
            all_masks[mask != 0] = mask[mask != 0]
    all_masks[all_masks == 0] = 255
    #print set(all_masks.flatten())

    out_dir = 'labels_{}'.format(dataType)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_png = os.path.join(out_dir, img['file_name'].replace('.jpg', '.png'))
    print('Saving label to {}\n\tand vis to {}'.format(out_png, out_png.replace('.png', '_vis.png')))

    all_masks_vis = all_masks.copy()
    all_masks_vis[all_masks_vis==255] = 0
    im=plt.imshow(all_masks_vis)
    ### Modified from https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib ###
    values = list(set(all_masks_vis.flatten()))
    # get the colors of the values, according to the colormap used by imshow
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    try:
        label_to_name = coco_id_to_name
        label_to_name[0] = 'BACKGROUND'
        patches = [ mpatches.Patch(color=colors[i], label="{}".format(label_to_name[values[i]]) ) for i in range(len(values)) ]
    except Exception as e:
        print e
        import ipdb; ipdb.set_trace()
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    #########################################################################################################

    plt.savefig(out_png.replace('.png', '_vis.png'), bbox_inches='tight')
    PIL.Image.fromarray(all_masks).save(out_png)
