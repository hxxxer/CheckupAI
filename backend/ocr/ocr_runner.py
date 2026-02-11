import sys
import os
import paddle
from paddleocr import PaddleOCRVL
from backend.config import settings


def process(file_path: str):
    device = "gpu:" + str(settings.ocr_gpu_id) if settings.ocr_use_gpu else "cpu"

    ocr_pipeline = PaddleOCRVL(
        # layout_detection_model_dir="/root/autodl-tmp/models/Paddle/PP-DocLayoutV3",
        # vl_rec_model_dir="/root/autodl-tmp/models/Paddle/PaddleOCR-VL-1.5"
        device=device,
        # show_log=False
    )

    # 执行OCR
    output = ocr_pipeline.predict_iter(file_path)

    return output


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    file_path = sys.argv[1]
    output_json_path = sys.argv[2]

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    output = process(file_path)

    for res in output:
        res.save_to_json(save_path=output_json_path)

    print("保存成功，路径：", output_json_path)
