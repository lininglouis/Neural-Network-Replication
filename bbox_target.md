        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])
