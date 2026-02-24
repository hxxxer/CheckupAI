"""
PaddleOCR Runner Script
被 subprocess 调用的 OCR 执行脚本
"""

import os
import argparse
from datetime import datetime
from paddleocr import PaddleOCRVL


def process(file_path: str, device: str = "cpu"):
    """
    执行 OCR 处理
    
    Args:
        file_path: 输入图片路径
        use_gpu: 是否使用 GPU
        gpu_id: GPU 设备 ID
        
    Returns:
        OCR 输出结果
    """
    
    ocr_pipeline = PaddleOCRVL(
        device=device,
    )
    
    # 执行 OCR
    output = ocr_pipeline.predict_iter(file_path)
    
    return output


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR Runner")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出 JSON 目录")
    parser.add_argument("--gpu", action="store_true", help="是否使用 GPU")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU 设备 ID")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 执行 OCR
    device = f"gpu:{args.gpu_id}" if args.gpu else "cpu"
    output = process(args.image, device=device)
    
    # 保存结果
    for res in output:
        # 创建以输入文件名命名的子目录
        base_name = os.path.basename(res['input_path'])
        # 添加页面索引作为更深层的目录
        page_index = res.get('page_index', 0)
        output_subdir = os.path.join(args.output, base_name, str(page_index))
        os.makedirs(output_subdir, exist_ok=True)
        
        # 保存 JSON 结果到子目录
        res.save_to_json(save_path=output_subdir)
        
        # 保存图片到子目录
        for idx, img_info in enumerate(res['imgs_in_doc']):
            img = img_info['img']
            save_path = os.path.join(output_subdir, img_info['path'])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
    
    print(f"保存成功，路径：{args.output}")


if __name__ == "__main__":
    main()
