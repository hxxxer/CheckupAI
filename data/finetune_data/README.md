# Sample Fine-tuning Data Format

This directory contains training data for LoRA fine-tuning.

## Data Format

Each JSON entry should have the following structure:

```json
{
  "text": "Instruction: 分析以下体检报告\n\n体检数据:\n- 血压: 140/90 mmHg (参考范围: <120/80)\n- 血糖: 6.5 mmol/L (参考范围: 3.9-6.1)\n\n回答:\n根据提供的体检数据,发现以下异常:\n1. 血压偏高,达到140/90 mmHg,超出正常参考范围(<120/80),提示可能存在高血压风险\n2. 血糖水平为6.5 mmol/L,超过正常参考范围上限(6.1),属于空腹血糖受损状态\n\n建议:\n- 血压方面:建议改善生活方式,减少钠盐摄入,增加运动,如果持续偏高需就医\n- 血糖方面:需要控制饮食,减少高糖食物摄入,定期监测血糖,必要时咨询内分泌科医生"
}
```

## Sample Data

Create `train.json` file with your training examples.
