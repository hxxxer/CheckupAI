import os

from paddleocr import PaddleOCRVL

input_path = "./raw"
output_path = "./processed"
os.makedirs(output_path, exist_ok=True)

pipeline = PaddleOCRVL()

output = pipeline.predict(input=input_path)

pages_res = list(output)

# output = pipeline.restructure_pages(pages_res, merge_tables=True, relevel_titles=True) # 合并跨页表格，重建多级标题
output = pipeline.restructure_pages(pages_res, merge_tables=True, relevel_titles=True, concatenate_pages=True) # 合并跨页表格，重建多级标题，合并多页结果为一页

for res in output:
    # res.print()
    # res.save_to_json(save_path=output_path)
    res.save_to_markdown(save_path=output_path)