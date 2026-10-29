#!/usr/bin/env python3
"""园头村快速补推：复用完整 pre 滑窗网格，缓存 >=0.05 候选，按阈值导出 SHP。"""
import argparse

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import concurrent.futures
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.affinity import affine_transform
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

from detect_speedup import run_rfdetr
from detect_speedup.building_vector_export import polygons_only, write_vectors
from detect_speedup.yolo_image_io_and_export import generate_starts
from run_building_best_pre import SOURCE, TRAIN, MODELS

OLD=Path('/data1/wzh26/重研/outputs/pre_20261028_rfdetr_best')
NAMES={'bonai':'BONAI','cn_typical':'中国典型建筑物'}


def prepare_crop(name, root, bounds):
    spec=MODELS[name];size,stride=spec['tile_size'],spec['stride']
    target=root/'cache'/('pre_'+name+'.tif')
    with rasterio.open(SOURCE) as src:
        roi=from_bounds(*bounds,transform=src.transform)
        xs=[x for x in generate_starts(src.width,size,stride)
            if x+size>roi.col_off-size and x<roi.col_off+roi.width+size]
        ys=[y for y in generate_starts(src.height,size,stride)
            if y+size>roi.row_off-size and y<roi.row_off+roi.height+size]
        left,top=min(xs),min(ys)
        width,height=max(xs)+size-left,max(ys)+size-top
        width,height=min(width,src.width-left),min(height,src.height-top)
        assert [x+left for x in generate_starts(width,size,stride)]==xs
        assert [y+top for y in generate_starts(height,size,stride)]==ys
        window=Window(left,top,width,height)
        profile=src.profile.copy()
        profile.update(driver='GTiff',width=width,height=height,transform=src.window_transform(window),
                       tiled=True,blockxsize=256,blockysize=256,compress='LZW')
        if not target.exists():
            with rasterio.open(target,'w',**profile) as dst:dst.write(src.read(window=window))
    print('%s: 原网格对应区域 %dx%d，共%d个切片'%(name,width,height,len(xs)*len(ys)),flush=True)
    return target, len(xs)*len(ys)


def export(name, root, threshold, frame, valid):
    raw=root/name/'raw'/(('pre_'+name)+'.json')
    data=json.loads(raw.read_text(encoding='utf-8'))
    assert data['conf']==.05 and threshold>=data['conf']
    with rasterio.open(root/'cache'/('pre_'+name+'.tif')) as src:
        a=src.transform
    to_post=Transformer.from_crs(4326,4549,always_xy=True).transform
    records={'图幅范围':[],'有效影像范围':[]}
    for i,det in enumerate(data['detections'],1):
        if det['score']<threshold:continue
        pixel=shape(det['mask_geometry'])
        world=affine_transform(pixel,[a.a,a.b,a.d,a.e,a.c,a.f])
        projected=transform(to_post,world)
        for scope,footprint in [('图幅范围',frame),('有效影像范围',valid)]:
            g=polygons_only(projected.intersection(footprint))
            if g is None or g.area<=0:continue
            assert footprint.buffer(1e-6).covers(g)
            props=dict(inst_id=i,score=float(det['score']),model=name,
                       area_m2=float(g.area),clipped=int(not footprint.covers(projected)))
            records[scope].append((g,props))
    tag='conf%03d'%round(threshold*100)
    directory=root/'exports'/tag/name;directory.mkdir(parents=True,exist_ok=True)
    for scope,features in records.items():
        write_vectors(directory/('园头村_'+scope+'_4549.shp'),features,'EPSG:4549','ESRI Shapefile',region=True)
    archive=directory.parent/(NAMES[name]+'_园头村_'+tag+'_EPSG4549.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as output:
        for p in directory.iterdir():
            if p.suffix in ['.shp','.shx','.dbf','.prj','.cpg']:output.write(p,p.name)
        output.writestr('README.txt','模型: %s\n置信度阈值: %.2f\nSHP坐标系: EPSG:4549\n仅推理pre.tif的园头村对应区域及周边上下文，保持原滑窗位置。\n'%(NAMES[name],threshold))
    with zipfile.ZipFile(archive) as output:assert output.testzip() is None
    result=dict(model=name,threshold=threshold,crs='EPSG:4549',
                frame_instances=len(records['图幅范围']),valid_instances=len(records['有效影像范围']),archive=str(archive))
    (directory/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('已导出',result,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold',type=float,default=.2)
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('/data1/wzh26/重研/outputs/园头村_rfdetr_candidates'))
    args=parser.parse_args()
    if not .05<=args.threshold<=1:parser.error('缓存支持 0.05 至 1.0 的导出阈值')
    root=args.output;root.mkdir(parents=True,exist_ok=True);(root/'cache').mkdir(exist_ok=True)
    info=json.loads((OLD/'shared/region_metadata.json').read_text(encoding='utf-8'))
    features=json.loads((OLD/'shared/园头村_范围_4549.geojson').read_text(encoding='utf-8'))['features']
    frame,valid=[shape(f['geometry']) for f in features]
    if not args.export_only:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            crops=dict(zip(MODELS,pool.map(lambda name:prepare_crop(name,root,info['frame_wgs84_bounds']),MODELS)))
        for name,spec in MODELS.items():
            crop,total=crops[name]
            directory=root/name/'raw';directory.mkdir(parents=True,exist_ok=True)
            raw=directory/(crop.stem+'.json')
            if not raw.exists():
                run_rfdetr(common=dict(input_path=str(crop),output_dir=str(directory),devices=['cuda:0'],
                    worker_count=1,reader_workers=2,batch_size=2,tile_size=spec['tile_size'],stride=spec['stride'],
                    save_json=True,save_bbox_shp=False,save_mask_shp=False,global_nms=True,global_iou=.45,
                    classes=[0],input_color='rgb',fp16=True,regularize=None),
                    rfdetr=dict(checkpoint=str(TRAIN/name/'rfdetr_seg_m/best.pth'),num_classes=1,
                                resolution=spec['resolution'],conf_threshold=.05,iou_threshold=.5))
            data=json.loads(raw.read_text(encoding='utf-8'))
            assert data['tiles']==data['processed_tiles']==total
            export(name,root,args.threshold,frame,valid)
    else:
        for name in MODELS:export(name,root,args.threshold,frame,valid)
    (root/'status.json').write_text(json.dumps({'state':'completed','threshold':args.threshold,'candidate_min_score':.05,
        'scope':'园头村对应pre区域及上下文，不是完整pre.tif','region_crs':'EPSG:4549'},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
