"""建筑实例矢量导出、跨 CRS 区域裁切与轻量预览，不修改源影像。"""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fiona
import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform, unary_union
from shapely import make_valid


def polygons_only(geometry):
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if geometry.geom_type == 'Polygon':
        return MultiPolygon([geometry])
    if geometry.geom_type == 'MultiPolygon':
        return geometry
    if geometry.geom_type == 'GeometryCollection':
        parts = []
        for part in geometry.geoms:
            valid = polygons_only(part)
            if valid is not None:
                parts.extend(valid.geoms)
        return MultiPolygon(parts) if parts else None
    return None


def preview(image, output, max_side):
    with rasterio.open(image) as src:
        scale = max(1., max(src.width, src.height) / max_side)
        width, height = math.ceil(src.width/scale), math.ceil(src.height/scale)
        array = src.read([1, 2, 3], out_shape=(3, height, width), resampling=Resampling.bilinear)
        Image.fromarray(array.transpose(1, 2, 0)).save(output)
        affine = src.transform * Affine.scale(src.width/width, src.height/height)
        return affine, (width, height)


def prepare_region(pre, post, root, post_crs='EPSG:4549'):
    import rasterio.features
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with rasterio.open(pre) as src:
        pre_crs, pre_affine = src.crs, src.transform
        pre_frame = Polygon([src.transform*p for p in [(0,0),(src.width,0),(src.width,src.height),(0,src.height)]])
    with rasterio.open(post) as src:
        if src.crs is not None and CRS(src.crs) != CRS(post_crs):
            raise ValueError('ROI 文件坐标系与指定坐标系不一致')
        post_frame = Polygon([src.transform*p for p in [(0,0),(src.width,0),(src.width,src.height),(0,src.height)]])
        width = min(src.width, 4096)
        height = math.ceil(src.height * width / src.width)
        mask = src.dataset_mask(out_shape=(height, width), resampling=Resampling.nearest) > 0
        mask_affine = src.transform * Affine.scale(src.width/width, src.height/height)
        valid = unary_union([shape(g) for g, value in rasterio.features.shapes(
            mask.astype(np.uint8), mask=mask, transform=mask_affine) if value == 1])
        valid = polygons_only(valid.intersection(post_frame))
        if valid is None:
            raise RuntimeError('ROI 没有有效影像覆盖范围')
        mask_resolution = [abs(mask_affine.a), abs(mask_affine.e)]
    to_pre = Transformer.from_crs(post_crs, pre_crs, always_xy=True).transform
    frame_pre = transform(to_pre, post_frame)
    if not pre_frame.buffer(1e-9).covers(frame_pre):
        raise ValueError('指定 ROI 不完整包含在 pre.tif 中，请检查坐标系')
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(preview, pre, root/'pre_original.jpg', 4096)
        b = pool.submit(preview, post, root/'园头村_original.jpg', 2048)
        pre_preview, pre_size = a.result()
        post_preview, post_size = b.result()
    info = {'pre_crs': pre_crs.to_string(), 'post_crs': post_crs,
            'post_crs_source': 'user confirmed EPSG:4549',
            'pre_affine': list(pre_affine)[:6], 'post_bounds': list(post_frame.bounds),
            'frame_wgs84_bounds': list(frame_pre.bounds),
            'valid_footprint_resolution_m': mask_resolution,
            'valid_footprint_approximate': True, 'pre_preview_size': pre_size, 'post_preview_size': post_size}
    (root/'region_metadata.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    (root/'园头村_范围_4549.geojson').write_text(json.dumps({
        'type':'FeatureCollection', 'crs':{'type':'name','properties':{'name':post_crs}},
        'features':[{'type':'Feature','properties':{'scope':'frame'},'geometry':mapping(post_frame)},
                    {'type':'Feature','properties':{'scope':'valid_data_approx'},'geometry':mapping(valid)}]
    }, ensure_ascii=False), encoding='utf-8')
    return dict(info=info, frame=post_frame, valid=valid, frame_pre=frame_pre,
                pre_affine=pre_affine, pre_preview=pre_preview, post_preview=post_preview)


def write_vectors(path, records, crs, driver, layer=None, region=False):
    schema = {'geometry':'MultiPolygon', 'properties':{'inst_id':'int', 'score':'float', 'model':'str:16'}}
    if region:
        schema['properties'].update(area_m2='float', clipped='int')
    options = dict(driver=driver, schema=schema, crs_wkt=CRS(crs).to_wkt(), encoding='UTF-8')
    if layer:
        options['layer'] = layer
    if driver == 'GeoJSON':
        options['RFC7946'] = 'YES'
    with fiona.open(path, 'w', **options) as output:
        output.writerecords({'geometry':mapping(g), 'properties':properties} for g, properties in records)
    with fiona.open(path, layer=layer) as source:
        if len(source) != len(records):
            raise RuntimeError('矢量数量校验失败: ' + str(path))


def draw_overlay(background, records, affine, output):
    image = Image.open(background).convert('RGBA')
    layer = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(layer)
    inv = ~affine
    for geometry, _ in records:
        for poly in geometry.geoms:
            points = [inv*(x,y) for x,y in poly.exterior.coords]
            draw.polygon(points, fill=(30,230,130,65), outline=(0,255,90,230), width=2)
            for hole in poly.interiors:
                draw.polygon([inv*(x,y) for x,y in hole.coords], fill=(0,0,0,0))
    Image.alpha_composite(image, layer).convert('RGB').save(output, quality=94)


def export_predictions(json_path, model_name, output, shared, region):
    output, shared = Path(output), Path(shared)
    output.mkdir(parents=True, exist_ok=True)
    document = json.loads(Path(json_path).read_text(encoding='utf-8'))
    source_crs = region['info']['pre_crs']
    post_crs = region['info']['post_crs']
    to_post = Transformer.from_crs(source_crs, post_crs, always_xy=True).transform
    to_wgs = Transformer.from_crs(post_crs, 'EPSG:4326', always_xy=True).transform
    a = region['pre_affine']
    affine = [a.a, a.b, a.d, a.e, a.c, a.f]
    full, frame_records, valid_records = [], [], []
    invalid = 0
    for i, detection in enumerate(document['detections'], 1):
        raw = detection.get('mask_geometry')
        if raw is None:
            raise ValueError('检测结果缺少实例掩膜，拒绝以框替代')
        pixel = polygons_only(shape(raw))
        if pixel is None:
            invalid += 1
            continue
        world = polygons_only(affine_transform(pixel, affine))
        properties = dict(inst_id=i, score=float(detection['score']), model=model_name)
        full.append((world, properties))
        if not world.intersects(region['frame_pre']):
            continue
        projected = transform(to_post, world)
        for footprint, records in [(region['frame'], frame_records), (region['valid'], valid_records)]:
            clipped = polygons_only(projected.intersection(footprint))
            if clipped is None or clipped.area <= 0:
                continue
            if not footprint.buffer(1e-6).covers(clipped):
                raise RuntimeError('区域裁切的范围校验失败')
            values = dict(properties, area_m2=float(clipped.area), clipped=int(not footprint.covers(projected)))
            records.append((clipped, values))
    if not full:
        raise RuntimeError('推理没有导出任何有效实例掩膜')
    gpkg = output/'building_instances.gpkg'
    if gpkg.exists():
        gpkg.unlink()
    write_vectors(gpkg, full, source_crs, 'GPKG', layer='pre_instances')
    write_vectors(gpkg, frame_records, post_crs, 'GPKG', layer='yuantou_frame', region=True)
    write_vectors(gpkg, valid_records, post_crs, 'GPKG', layer='yuantou_valid', region=True)
    if set(fiona.listlayers(gpkg)) != {'pre_instances', 'yuantou_frame', 'yuantou_valid'}:
        raise RuntimeError('GeoPackage 图层完整性校验失败')
    write_vectors(output/'pre_buildings.shp', full, source_crs, 'ESRI Shapefile')
    write_vectors(output/'园头村_图幅范围_4549.shp', frame_records, post_crs, 'ESRI Shapefile', region=True)
    write_vectors(output/'园头村_有效影像范围_4549.shp', valid_records, post_crs, 'ESRI Shapefile', region=True)
    write_vectors(output/'园头村_WGS84.geojson', [(transform(to_wgs,g),p) for g,p in valid_records],
                  'EPSG:4326', 'GeoJSON', region=True)
    draw_overlay(shared/'pre_original.jpg', full, region['pre_preview'], output/'pre_overlay.jpg')
    draw_overlay(shared/'园头村_original.jpg', valid_records, region['post_preview'], output/'园头村_overlay.jpg')
    summary = dict(model=model_name, full_instances=len(full), frame_instances=len(frame_records),
                   valid_region_instances=len(valid_records), invalid_instances=invalid,
                   confidence=document['conf'], tiles=document['tiles'], resolution=document['resolution'],
                   source_pre_crs=source_crs, region_crs=post_crs,
                   valid_footprint_resolution_m=region['info']['valid_footprint_resolution_m'])
    (output/'vector_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('矢量与区域裁切完成: %s' % summary, flush=True)
    return summary


def comparison(shared, models, output):
    shared, output = Path(shared), Path(output)
    for stem, background, model_image in [('pre_compare','pre_original.jpg','pre_overlay.jpg'),
                                           ('园头村_compare','园头村_original.jpg','园头村_overlay.jpg')]:
        sources = [('Original', shared/background)] + [(name,Path(directory)/model_image) for name,directory in models]
        images = []
        for name, path in sources:
            image = Image.open(path).convert('RGB')
            image.thumbnail((1800,1600))
            labeled = Image.new('RGB', (image.width,image.height+40), '#101827')
            labeled.paste(image,(0,40));ImageDraw.Draw(labeled).text((12,12),name,fill='white')
            images.append(labeled)
        result = Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)),'#101827')
        x=0
        for image in images:
            result.paste(image,(x,0));x+=image.width
        result.save(output/(stem+'.jpg'),quality=94)
