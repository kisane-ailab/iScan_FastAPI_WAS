import numpy as np

def nms(boxes, scores, overlap_threshold=0.5):
    if len(boxes) == 0:
        return []

    # Extract separate coordinates and scores
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_indices = scores.argsort()[::-1]

    keep = []
    while sorted_indices.size > 0:
        i = sorted_indices[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[sorted_indices[1:]])
        yy1 = np.maximum(y1[i], y1[sorted_indices[1:]])
        xx2 = np.minimum(x2[i], x2[sorted_indices[1:]])
        yy2 = np.minimum(y2[i], y2[sorted_indices[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / (areas[i] + areas[sorted_indices[1:]] - (w * h))
        sorted_indices = sorted_indices[np.where(overlap <= overlap_threshold)[0] + 1]

    return keep

if __name__=='__main__':
    # Example usage
    boxes_with_scores = np.array([
        [50, 50, 150, 150, 0.9],
        [55, 60, 155, 160, 0.8],
        [200, 200, 300, 300, 0.75],
    ])

    selected_indices = nms(boxes_with_scores[:, :4], boxes_with_scores[:, 4])
    selected_boxes = boxes_with_scores[selected_indices]