#python demo_kiat.py --src_dir=/home/ayca/kiat/ROCA_demo/imgs --res_dir=/home/ayca/kiat/ROCA_demo/res --model_path=/home/ayca/kiat/ROCA_files/Models/model_best.pth --data_dir=/home/ayca/kiat/ROCA_files/Data/Dataset --config_path=/home/ayca/kiat/ROCA_files/Models/config.yaml

import os
import sys
import pdb
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from trimesh.exchange.export import export_mesh
from trimesh.util import concatenate as stack_meshes

from roca.engine import Predictor


def main(args):
    wild = True
    predictor = Predictor(
        data_dir=args.data_dir,
        model_path=args.model_path,
        config_path=args.config_path,
        wild=wild #args.wild,
    )
    to_file = args.res_dir != 'none'
    if to_file:
        os.makedirs(args.res_dir, exist_ok=True)

    #pdb.set_trace()
    img_names = [el for el in os.listdir(args.src_dir) if (el.split('.')[-1]=='jpg' or el.split('.')[-1]=='jpeg' or el.split('.')[-1]=='png')]
    for img_name in img_names:
        img_path = os.path.join(args.src_dir, img_name)
        img_orig = Image.open(img_path)
        img = img_orig.resize((480,360)) #add an assertion here before applying this transformation
        # resize the image?
        #pdb.set_trace()
        img = np.asarray(img)
        instances, cad_ids = predictor(img) #, scene=""
        meshes = predictor.output_to_mesh(
            instances,
            cad_ids,
            # Table works poorly in the wild case due to size diversity
            excluded_classes={'table'} if wild else (), #maybe don't exclude?
            as_open3d=not to_file
        )

        if predictor.can_render:
            rendering, ids = predictor.render_meshes(meshes)
            mask = ids > 0
            overlay = img.copy()
            overlay[mask] = np.clip(
                0.8 * rendering[mask] * 255 + 0.2 * overlay[mask], 0, 255
            ).astype(np.uint8)
            if to_file:
                Image.fromarray(overlay).save(
                    os.path.join(args.res_dir, 'overlay_{}.jpg'.format(img_name))
                )
            else:
                img = o3d.geometry.Image(overlay)
                o3d.visualization.draw_geometries([img], height=480, width=640)

        if to_file:
            out_file = os.path.join(args.res_dir, 'mesh_{}.ply'.format(img_name))
            export_mesh(stack_meshes(meshes), out_file, file_type='ply')
        else:
            o3d.visualization.draw_geometries(meshes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--res_dir', default='none')
    args = parser.parse_args(sys.argv[1:])
    main(args)
