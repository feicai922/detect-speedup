import json
import tempfile
import unittest
from pathlib import Path
from affine import Affine
from PIL import Image
import fiona
from shapely.geometry import box, mapping, shape
from detect_speedup.building_vector_export import export_predictions


class RegionExportTests(unittest.TestCase):
    def test_instances_are_clipped_and_preserve_ids_and_projected_crs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root/'shared';shared.mkdir()
            for name in ['pre_original.jpg', '园头村_original.jpg']:
                Image.new('RGB',(10,10),'gray').save(shared/name)
            document = dict(conf=.3,tiles=1,resolution=504,detections=[
                dict(score=.9,mask_geometry=mapping(box(2,2,4,4))),
                dict(score=.8,mask_geometry=mapping(box(7,7,9,9))),
                dict(score=.7,mask_geometry=mapping(box(5,3,8,5)))])
            prediction=root/'pre.json';prediction.write_text(json.dumps(document),encoding='utf-8')
            region=dict(info=dict(pre_crs='EPSG:4549',post_crs='EPSG:4549',
                                 valid_footprint_resolution_m=[1,1]),
                        pre_affine=Affine.identity(),pre_preview=Affine.identity(),post_preview=Affine.identity(),
                        frame=box(1,1,9,9),valid=box(2,2,6,6),frame_pre=box(1,1,9,9))
            result=export_predictions(prediction,'test',root/'result',shared,region)
            self.assertEqual(result['full_instances'],3)
            self.assertEqual(result['frame_instances'],3)
            self.assertEqual(result['valid_region_instances'],2)
            with fiona.open(root/'result/building_instances.gpkg',layer='yuantou_valid') as src:
                self.assertEqual(src.crs.to_epsg(),4549)
                features=list(src)
                self.assertEqual([f['properties']['inst_id'] for f in features],[1,3])
                self.assertEqual(features[1]['properties']['clipped'],1)
                self.assertAlmostEqual(shape(features[1]['geometry']).area,2)


if __name__=='__main__':
    unittest.main()
