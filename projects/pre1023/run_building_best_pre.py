#!/usr/bin/env python3
"""使用 detect-speedup 对 pre.tif 推理两个 RF-DETR 最佳模型并裁切园头村。"""
import argparse

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

SOURCE = Path('/data1/wzh26/data/1023测试影像/1/pre.tif')
ROI = Path('/data1/wzh26/data/1023测试影像/1/post/园头村/res_ortho_tdom.tif')
TRAIN = Path('/data1/wzh26/重研/code/runs')
MODELS = {'bonai':dict(resolution=1008,tile_size=1024,stride=768),
          'cn_typical':dict(resolution=504,tile_size=512,stride=384)}
STATUS_PATH = None


def write_status(path, **updates):
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    old.update(updates,updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def provenance(name):
    import torch
    path=TRAIN/name/'rfdetr_seg_m/best.pth'
    ckpt=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert ckpt['model_name']=='RFDETRSegMedium'
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(4*1024*1024),b''):digest.update(chunk)
    return dict(path=str(path),epoch=int(ckpt['epoch'])+1,sha256=digest.hexdigest(),
                resolution=MODELS[name]['resolution'],model_name=ckpt['model_name'])


def smoke(name, manifest, root):
    import torch
    import rasterio
    from rasterio.windows import Window
    from detect_speedup.rfdetr_large_image_inference import MultiprocessRFDETRInferConfig,load_reusable_model,predict_batch
    spec=MODELS[name]
    cfg=MultiprocessRFDETRInferConfig(source=str(SOURCE),checkpoint=manifest['path'],resolution=spec['resolution'],
              device='cuda:0',devices=['cuda:0'],batch_size=2,tile_size=spec['tile_size'],input_color='rgb',fp16=True,
              conf_threshold=.3,classes=[0])
    size=spec['tile_size'];window=(3500,18200,3500+size,18200+size)
    with rasterio.open(SOURCE) as src:
        image=src.read(window=Window(window[0],window[1],size,size)).transpose(1,2,0)
    model=load_reusable_model(cfg,'cuda:0')
    prediction=predict_batch(model,[image],[window],19795,20748,cfg)
    assert all(d['mask_geometry'] for d in prediction)
    report=dict(passed=True,model=name,detections=len(prediction),window=window,
                loaded_model=type(model.model).__name__)
    (root/(name+'_smoke.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('GPU smoke:',report,flush=True)
    del model;torch.cuda.empty_cache()


def main():
    global STATUS_PATH
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/data1/wzh26/重研/outputs/pre_20261028_rfdetr_best'))
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    STATUS_PATH = root/'status.json'
    import torch
    import rasterio
    assert os.environ.get('CUDA_DEVICE_ORDER')=='FASTEST_FIRST' and os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    assert torch.cuda.device_count()==1 and 'A6000' in torch.cuda.get_device_name(0)
    from detect_speedup import run_rfdetr
    from detect_speedup.yolo_image_io_and_export import count_windows
    from detect_speedup.building_vector_export import prepare_region,export_predictions,comparison
    write_status(root/'status.json',state='preparing',gpu_node='gpu7',gpu_index=7,source=str(SOURCE),roi=str(ROI),roi_crs='EPSG:4549',error=None,errors=[])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        manifests=dict(zip(MODELS,pool.map(provenance,MODELS)))
    (root/'model_provenance.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in MODELS:
        smoke(name,manifests[name],root)
    if args.smoke_only:
        write_status(root/'status.json',state='smoke_passed');return
    shared=root/'shared';region=prepare_region(SOURCE,ROI,shared,'EPSG:4549')
    exports=[];errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as export_pool:
        for name,spec in MODELS.items():
            directory=root/name;directory.mkdir(exist_ok=True)
            common=dict(input_path=str(SOURCE),output_dir=str(directory/'json'),devices=['cuda:0'],worker_count=1,
                        reader_workers=4,queue_size=4,batch_size=2,tile_size=spec['tile_size'],stride=spec['stride'],
                        save_json=True,save_bbox_shp=False,save_mask_shp=False,global_nms=True,global_iou=.45,
                        classes=[0],input_color='rgb',fp16=True,regularize=None)
            backend=dict(checkpoint=manifests[name]['path'],num_classes=1,resolution=spec['resolution'],
                         conf_threshold=.3,iou_threshold=.5)
            write_status(root/'status.json',state='inference',current_model=name)
            prediction_path=directory/'json/pre.json'
            for attempt in range(2):
                try:
                    run_rfdetr(common=common,rfdetr=backend)
                    document=json.loads(prediction_path.read_text(encoding='utf-8'))
                    with rasterio.open(SOURCE) as src:
                        assert document['tiles']==count_windows(src.width,src.height,spec['tile_size'],spec['stride'])
                    assert document['processed_tiles']==document['tiles']
                    assert document['instance_segmentation'] and document['merged_detections']==len(document['detections'])
                    assert all(d.get('mask_geometry') for d in document['detections'])
                    exports.append((name,export_pool.submit(export_predictions,prediction_path,name,directory,shared,region)))
                    break
                except Exception as exc:
                    print('模型 %s 第%d次失败: %s'%(name,attempt+1,repr(exc)),flush=True)
                    if attempt==1:errors.append(dict(model=name,error=repr(exc)))
                    else:time.sleep(2)
        summaries=[]
        for name,job in exports:
            try:summaries.append(job.result())
            except Exception as exc:errors.append(dict(model=name,error=repr(exc)))
    if errors:
        write_status(root/'status.json',state='failed',errors=errors,results=summaries)
        raise RuntimeError(str(errors))
    comparison(shared,[(name,root/name) for name in MODELS],root)
    write_status(root/'status.json',state='completed',current_model=None,results=summaries,errors=[])
    print('全部推理、矢量裁切和可视化完成。',flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if STATUS_PATH is not None:
            write_status(STATUS_PATH,state='failed',error=repr(exc))
        raise
