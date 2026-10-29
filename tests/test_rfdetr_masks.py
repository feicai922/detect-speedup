import unittest
import numpy as np
import torch
from shapely.geometry import shape
from detect_speedup.rfdetr_large_image_inference import mask_geometry, _convert_tile_predictions


class MaskGeometryTests(unittest.TestCase):
    def test_parallel_masks_match_single_instance_results(self):
        masks = torch.zeros(16,1,20,20,dtype=torch.bool)
        masks[:,:,2:14,3:15] = True
        masks[:,:,5:8,6:9] = False
        prediction = dict(boxes=torch.tensor([[3.,2.,15.,14.]]*16),
                          scores=torch.linspace(.4,.9,16),labels=torch.zeros(16,dtype=torch.long),masks=masks)
        bulk = _convert_tile_predictions(prediction,(100,200,120,220),500,500,['building'],.3,[0])
        sequential = []
        for i in range(16):
            item = {k:v[i:i+1] for k,v in prediction.items()}
            sequential.extend(_convert_tile_predictions(item,(100,200,120,220),500,500,['building'],.3,[0]))
        self.assertEqual(bulk,sequential)

    def test_rfdetr_scripted_losses_import_after_framework(self):
        from rfdetr.utilities.box_ops import batch_sigmoid_ce_loss_jit
        self.assertIsInstance(batch_sigmoid_ce_loss_jit, torch.jit.ScriptFunction)

    def test_holes_components_and_tile_offsets(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:12, 3:13] = True
        mask[5:8, 6:9] = False
        mask[16:18, 16:19] = True
        primary, geometry = mask_geometry(mask, 100, 200)
        poly = shape(geometry)
        self.assertEqual(poly.geom_type, 'MultiPolygon')
        self.assertEqual(poly.area, mask.sum())
        self.assertEqual(poly.bounds, (103., 202., 119., 218.))
        self.assertEqual(len(primary), 5)

    def test_segmentation_is_exported_and_low_scores_are_filtered(self):
        masks = torch.zeros(2, 1, 20, 20, dtype=torch.bool)
        masks[:, :, 2:6, 4:8] = True
        pred = dict(boxes=torch.tensor([[4.,2.,8.,6.], [4.,2.,8.,6.]]),
                    scores=torch.tensor([.8, .1]), labels=torch.tensor([0,0]), masks=masks)
        result = _convert_tile_predictions(pred, (100,200,120,220), 500,500, ['building'],.3,[0])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bbox_xyxy'], [104.,202.,108.,206.])
        self.assertEqual(shape(result[0]['mask_geometry']).area, 16)


if __name__ == '__main__':
    unittest.main()
