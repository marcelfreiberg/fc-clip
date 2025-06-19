import torch
import json
import os
import csv
import argparse
from pathlib import Path
from tabulate import tabulate
from datetime import datetime


def save_results_to_files(results_data, output_dir, eval_type, tag="P0"):
    """Save evaluation results to multiple file formats"""
    
    # Create output directory for saved results
    save_dir = Path(output_dir) / "saved_metrics" / tag / eval_type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to JSON (machine readable)
    json_file = save_dir / f"{eval_type}_metrics_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Saved JSON results to: {json_file}")
    
    # Save per-class results to CSV (spreadsheet friendly)
    if 'per_class_metrics' in results_data:
        csv_file = save_dir / f"{eval_type}_per_class_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if eval_type == 'semantic':
                writer.writerow(['Class Name', 'IoU (%)', 'Accuracy (%)'])
                for class_data in results_data['per_class_metrics']:
                    writer.writerow(class_data)
            elif eval_type == 'panoptic':
                writer.writerow(['Category ID', 'Category Name', 'Segment Count'])
                for class_data in results_data['per_class_metrics']:
                    writer.writerow(class_data)
                    
        print(f"✓ Saved CSV results to: {csv_file}")
    
    # Save formatted text report (human readable)
    txt_file = save_dir / f"{eval_type}_report_{timestamp}.txt"
    with open(txt_file, 'w') as f:
        f.write(f"FC-CLIP Evaluation Report - {eval_type.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tag: {tag}\n")
        f.write("="*60 + "\n\n")
        
        if 'overall_metrics' in results_data:
            f.write("OVERALL METRICS:\n")
            for metric, value in results_data['overall_metrics'].items():
                f.write(f"  {metric}: {value}\n")
            f.write("\n")
        
        if 'per_class_metrics' in results_data:
            f.write("PER-CLASS METRICS:\n")
            if eval_type == 'semantic':
                f.write(f"{'Class Name':<30} {'IoU (%)':<10} {'Accuracy (%)':<15}\n")
                f.write("-" * 60 + "\n")
                for class_data in results_data['per_class_metrics']:
                    f.write(f"{class_data[0]:<30} {class_data[1]:<10} {class_data[2]:<15}\n")
            elif eval_type == 'panoptic':
                f.write(f"{'Category':<20} {'Name':<25} {'Segments':<15}\n")
                f.write("-" * 65 + "\n")
                for class_data in results_data['per_class_metrics']:
                    f.write(f"{class_data[0]:<20} {class_data[1]:<25} {class_data[2]:<15}\n")
    
    print(f"✓ Saved text report to: {txt_file}")


def load_and_display_semantic_metrics(eval_file_path, save_results=True, tag="P0"):
    """Load and display per-class semantic segmentation metrics"""
    print(f"\n=== SEMANTIC SEGMENTATION METRICS ===")
    print(f"Loading from: {eval_file_path}")
    
    try:
        results = torch.load(eval_file_path, map_location='cpu')
        
        # Extract overall metrics
        overall_metrics = {
            'mIoU': results.get('mIoU', 0),
            'fwIoU': results.get('fwIoU', 0), 
            'mACC': results.get('mACC', 0),
            'pACC': results.get('pACC', 0)
        }
        
        # Extract per-class IoU and accuracy
        per_class_data = []
        class_names = []
        
        for key, value in results.items():
            if key.startswith('IoU-'):
                class_name = key[4:]  # Remove 'IoU-' prefix
                class_names.append(class_name)
                
        # Build per-class table
        for class_name in sorted(class_names):
            iou_key = f'IoU-{class_name}'
            acc_key = f'ACC-{class_name}'
            
            iou_val = results.get(iou_key, 0)
            acc_val = results.get(acc_key, 0)
            
            per_class_data.append([
                class_name,
                f"{iou_val:.2f}%",
                f"{acc_val:.2f}%"
            ])
        
        # Display overall metrics
        print(f"\nOVERALL METRICS:")
        overall_table = [
            ['Metric', 'Value'],
            ['mIoU (Mean IoU)', f"{overall_metrics['mIoU']:.2f}%"],
            ['fwIoU (Frequency Weighted IoU)', f"{overall_metrics['fwIoU']:.2f}%"],
            ['mACC (Mean Accuracy)', f"{overall_metrics['mACC']:.2f}%"],
            ['pACC (Pixel Accuracy)', f"{overall_metrics['pACC']:.2f}%"]
        ]
        print(tabulate(overall_table, headers='firstrow', tablefmt='grid'))
        
        # Display per-class metrics
        print(f"\nPER-CLASS METRICS:")
        headers = ['Class Name', 'IoU (%)', 'Accuracy (%)']
        print(tabulate(per_class_data, headers=headers, tablefmt='grid'))
        
        # Save results if requested
        if save_results:
            results_data = {
                'evaluation_type': 'semantic_segmentation',
                'overall_metrics': overall_metrics,
                'per_class_metrics': per_class_data,
                'raw_results': {k: float(v) if isinstance(v, (int, float)) else v for k, v in results.items()}
            }
            save_results_to_files(results_data, "output", "semantic", tag)
        
        return results
        
    except Exception as e:
        print(f"Error loading semantic metrics: {e}")
        return None


def load_and_display_panoptic_metrics(predictions_file, save_results=True, tag="P0"):
    """Load and display panoptic segmentation metrics from predictions.json"""
    print(f"\n=== PANOPTIC SEGMENTATION METRICS ===")
    print(f"Loading from: {predictions_file}")
    
    try:
        with open(predictions_file, 'r') as f:
            data = json.load(f)
        
        # Extract basic info
        total_images = len(data.get('images', []))
        total_annotations = len(data.get('annotations', []))
        categories = data.get('categories', [])
        
        print(f"Total images: {total_images}")
        print(f"Total annotations: {total_annotations}")
        print(f"Number of categories: {len(categories)}")
        
        # Analyze segments by category
        category_segments = {}
        category_names = {cat['id']: cat['name'] for cat in categories}
        
        for annotation in data.get('annotations', []):
            for segment in annotation.get('segments_info', []):
                cat_id = segment.get('category_id')
                if cat_id not in category_segments:
                    category_segments[cat_id] = 0
                category_segments[cat_id] += 1
        
        # Build per-class data
        per_class_data = []
        for cat_id, count in sorted(category_segments.items()):
            cat_name = category_names.get(cat_id, f"Unknown_{cat_id}")
            per_class_data.append([cat_id, cat_name, count])
        
        # Display category breakdown
        print(f"\nCATEGORY BREAKDOWN:")
        headers = ['Category ID', 'Category Name', 'Segment Count']
        print(tabulate(per_class_data, headers=headers, tablefmt='grid'))
        
        # Display category definitions
        print(f"\nCATEGORY DEFINITIONS:")
        cat_table = []
        for cat in sorted(categories, key=lambda x: x['id']):
            cat_table.append([
                cat['id'], 
                cat['name'], 
                cat.get('supercategory', 'N/A'),
                'Thing' if cat.get('isthing') else 'Stuff'
            ])
        
        headers = ['ID', 'Name', 'Supercategory', 'Type']
        print(tabulate(cat_table, headers=headers, tablefmt='grid'))
        
        # Save results if requested
        if save_results:
            total_segments = sum(category_segments.values())
            results_data = {
                'evaluation_type': 'panoptic_segmentation',
                'total_images': total_images,
                'total_annotations': total_annotations,
                'total_segments': total_segments,
                'num_categories': len(categories),
                'per_class_metrics': per_class_data,
                'category_definitions': cat_table,
                'category_segments': category_segments
            }
            save_results_to_files(results_data, "output", "panoptic", tag)
        
        return data
        
    except Exception as e:
        print(f"Error loading panoptic metrics: {e}")
        return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract and save per-class metrics from FC-CLIP evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--tag", 
        type=str, 
        default="P0",
        help="Experiment tag to process (e.g., P0, P1, P2, etc.)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files, just display them"
    )
    
    parser.add_argument(
        "--eval-types",
        nargs="+",
        choices=["semantic", "panoptic"],
        default=["semantic", "panoptic"],
        help="Evaluation types to process"
    )
    
    return parser.parse_args()


def main():
    """Main function to extract metrics from all evaluation types"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Look for evaluation results in output directory
    output_dir = Path('output')
    
    if not output_dir.exists():
        print("No output directory found!")
        return
    
    print(f"🔍 Scanning for evaluation results...")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"🏷️  Processing experiment tag: {args.tag}")
    print(f"📊 Evaluation types: {', '.join(args.eval_types)}")
    print(f"💾 Save results: {not args.no_save}")
    
    # Process each evaluation type
    experiment_dir = output_dir / args.tag
    if not experiment_dir.exists():
        print(f"No experiment directory found for tag '{args.tag}'")
        return
        
    inference_dir = experiment_dir / 'inference'
    if not inference_dir.exists():
        print(f"No inference directory found for tag '{args.tag}'")
        return
    
    print(f"\n{'='*60}")
    print(f"PROCESSING FC-CLIP EVALUATION RESULTS (Tag: {args.tag})")
    print(f"{'='*60}")
    
    for eval_type in args.eval_types:
        print(f"\n{'-'*40}")
        print(f"Processing {eval_type.upper()} evaluation")
        print(f"{'-'*40}")
        
        if eval_type == 'semantic':
            # Load semantic segmentation metrics
            sem_eval_file = inference_dir / 'sem_seg_evaluation.pth'
            if sem_eval_file.exists():
                load_and_display_semantic_metrics(sem_eval_file, save_results=not args.no_save, tag=args.tag)
            else:
                print(f"No semantic evaluation file found at {sem_eval_file}")
                
        elif eval_type == 'panoptic':
            # Load panoptic predictions
            predictions_file = inference_dir / 'predictions.json'
            if predictions_file.exists():
                load_and_display_panoptic_metrics(predictions_file, save_results=not args.no_save, tag=args.tag)
            else:
                print(f"No panoptic predictions file found at {predictions_file}")
    
    if not args.no_save:
        print(f"\n🎉 Evaluation complete!")
        saved_dir = output_dir / "saved_metrics"
        if saved_dir.exists():
            print(f"📊 All metrics saved to: {saved_dir.absolute()}")
            print(f"   - JSON files for programmatic access")
            print(f"   - CSV files for spreadsheet analysis") 
            print(f"   - TXT files for human-readable reports")
    else:
        print(f"\n🎉 Evaluation complete! (Results not saved)")


if __name__ == '__main__':
    main() 