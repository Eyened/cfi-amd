import pandas as pd
import torch
from PIL import Image
import numpy as np
import os
import json
from utils.utils import open_image, to_uint8
from utils.report import Report
from utils.etdrs_masks import ETDRS_masks

from processor import Processor
from landmarks import LandmarksProcessor

feature_names = 'drusen', 'RPD', 'hyperpigmentation', 'rpe_degeneration'


def get_processors():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return Processor(device), LandmarksProcessor(device)


def export_features(result, base_path, export_probability, skip_empty=True):
    for feature_name in feature_names:
        if export_probability:
            img = Image.fromarray(result[feature_name])
        else:
            binary_image = result[feature_name] >= 0.5
            if skip_empty and not np.any(binary_image):
                continue
            img = Image.fromarray(to_uint8(binary_image))

        path = f'{base_path}/{feature_name}.png'
        img.save(path)
        print(f'exported {feature_name} to {path}')


def get_resolution(fovea_x, fovea_y, disc_x, disc_y):
    # estimate the resolution of the image

    # distance between fovea and disc edge
    dist = np.sqrt((fovea_x - disc_x) ** 2 + (fovea_y - disc_y) ** 2)

    # ETDRS grid radius, 3mm = 0.76 * dist between fovea and disc edge
    # This was empirically determined on Rotterdam Study data
    resolution = 3 / (0.76 * dist)

    return resolution


def get_etdrs_masks(bounds, coords):
    h, w = bounds.h, bounds.w
    fovea_x, fovea_y = coords['fovea']
    disc_x, disc_y = coords['disc_edge']
    resolution = get_resolution(fovea_x, fovea_y, disc_x, disc_y)
    laterality = 'R' if disc_x > fovea_x else 'L'

    return ETDRS_masks(
        h, w, fovea_x, fovea_y, resolution, laterality)


def export_results_full(output_folder, results):
    try:
        summary, bounds, coords = next(
            (summary, bounds, coords) for _, summary, bounds, coords
            in results if summary is not None)
    except StopIteration:
        print('No images were processed successfully')
        return

    keys = summary[feature_names[0]].keys()
    summary_header = [
        f'{feature_name}_{k}'
        for feature_name in feature_names
        for k in keys
    ]
    bounds_header = bounds.list_names
    coords_header = ["disc_edge_x", "disc_edge_y", "fovea_x", "fovea_y"]

    rows = []
    for row, summary, bounds, coords in results:
        row_out = [row.identifier, row.path]
        rows.append(row_out)
        if summary is None:
            row_out += [None] * len(summary_header)
        else:
            row_out += [
                summary[feature_name][k] for feature_name in feature_names for k in keys
            ]
        if bounds is None:
            row_out += [None] * len(bounds_header)
        else:
            row_out += bounds.to_list()
        if coords is None:
            row_out += [None] * len(coords_header)
        else:
            row_out += [*coords['disc_edge'], *coords['fovea']]

    pd.DataFrame(rows, columns=['identifier', 'path'] + summary_header + bounds_header + coords_header).to_csv(
        f'{output_folder}/results_full.csv', index=False)

def export_results_area(output_folder, results):
    keys = ['total_area', 'grid_area', 'outer_area', 'inner_area', 'center_area']
    summary_header = [
        f'{feature_name}_{k}'
        for feature_name in feature_names
        for k in keys
    ]
    
    rows = []
    for row, summary, _, _ in results:
        row_out = [row.identifier, row.path]
        rows.append(row_out)
        if summary is None:
            row_out += [None] * len(summary_header)
        else:
            row_out += [
                summary[feature_name][k] for feature_name in feature_names for k in keys
            ]
    
    pd.DataFrame(rows, columns=['identifier', 'path'] + summary_header).to_csv(
        f'{output_folder}/results_area.csv', index=False)


def main(csv_path, output_folder, args):
    print('Loading models...')
    processor, landmarksProcessor = get_processors()

    df = pd.read_csv(csv_path)

    results = []

    for idx, row in df.iterrows():
        print(f'Processing image {idx + 1}/{len(df)}')
        try:
            report, bounds, coords = process_row(
                output_folder, args, processor, landmarksProcessor, row)
            results.append((row, report.summaries, bounds, coords))
        except Exception as e:
            print(f'Error processing image {row.path}: {e}')
            results.append((row, None, None, None))

    export_results_full(output_folder, results)
    export_results_area(output_folder, results)


def process_row(output_folder, args, processor, landmarksProcessor, row):
    print(f'loading image {row.path}')
    image = open_image(row.path)

    result = processor.process(image)
    bounds = result['bounds']
    coords = landmarksProcessor.process(image, bounds)

    base_path = f'{output_folder}/{row.identifier}'
    os.makedirs(base_path, exist_ok=True)

    export_features(result, base_path,
                    args.export_probability, args.skip_empty)

    if args.export_bounds:
        with open(f'{base_path}/bounds.json', 'w') as f:
            json.dump(bounds.to_dict(), f)

    if args.export_coordinates:
        with open(f'{base_path}/coordinates.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in coords.items()}, f)

    etdrs_masks = get_etdrs_masks(bounds, coords)

    feature_images = {
        feature_name: result[feature_name] >= 0.5
        for feature_name in feature_names
    }
    report = Report(feature_images, etdrs_masks, etdrs_masks.all_fields)

    report.export(base_path, image, row.identifier,
                  args.export_html_report, True)

    return report, bounds, coords


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run inference on images listed in a CSV file.')
    parser.add_argument('--csv_path', type=str,
                        help='Path to the CSV file containing image paths.', default='/input.csv')
    parser.add_argument('--output_folder', type=str,
                        help='Folder to store the inference results.', default='/output')
    parser.add_argument('--export_probability', action=argparse.BooleanOptionalAction, default=False,
                        help='Export probability maps instead of binary masks')
    parser.add_argument('--skip_empty', action=argparse.BooleanOptionalAction, default=True,
                        help='Skip exporting empty segmentation masks')
    parser.add_argument('--export_html_report', action=argparse.BooleanOptionalAction, default=True,
                        help='Export HTML report with ETDRS grid and feature masks')
    parser.add_argument('--export_coordinates', action=argparse.BooleanOptionalAction, default=True,
                        help='Export coordinates of fovea and disc edge')
    parser.add_argument('--export_bounds', action=argparse.BooleanOptionalAction, default=True,
                        help='Export bounds of the image')

    args = parser.parse_args()
    main(args.csv_path, args.output_folder, args)
