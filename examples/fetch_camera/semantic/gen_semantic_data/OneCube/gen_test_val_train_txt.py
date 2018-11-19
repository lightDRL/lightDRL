
import glob


def gen_text( target_path, images_path , segs_path ,  prefix = '/SegNet/OneCube/' ):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
    images.sort()

    segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
    segmentations.sort()

    assert len( images ) == len(segmentations)

    f = open(target_path, "w")

    for im , seg in zip(images,segmentations):
        # print(f'im_fn = {im_fn}, seg_fn={seg_fn}')
        # print('  im_fn.split(/)[-1] =', im.split('/')[-1] )
        # print('  seg_fn.split(/)[-1] =', seg.split('/')[-1] )
        assert(  im.split('/')[-1] ==  seg.split('/')[-1] )

        s = "{}{} {}{}".format(prefix, im,prefix, seg)
        # print(s)
        f.write(s + '\n')

    f.close()


gen_text('train.txt', 'train/', 'trainannot/', prefix = '/SegNet/OneCube/' )
gen_text('val.txt', 'val/', 'valannot/', prefix = '/SegNet/OneCube/' )
gen_text('test.txt', 'test/', 'testannot/', prefix = '/SegNet/OneCube/' )
