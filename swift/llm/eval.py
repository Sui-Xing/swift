import os
import torch
from swift.llm import (
    get_template, inference, ModelType, get_default_template_type, get_model_tokenizer
)
from swift.utils import seed_everything
from swift.llm.utils.model import get_model_tokenizer_yi_vl
from datetime import datetime
import argparse
from rouge import Rouge
import time
from tqdm.auto import tqdm
from swift.tuners import Swift
import json
import pandas as pd
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 单个推理
def single_infer(template, model, system, query, images, history=None):
    if images != None:
        response, history = inference(model=model,
                                      template=template,
                                      query=query,
                                      history=history,
                                      system=system,
                                      images=images)
    else:
        response, history = inference(model=model,
                                      template=template,
                                      query=query,
                                      history=history,
                                      system=system,)
    return response


def get_output_by_image(error_images, error_response):
    query_out = ''
    for i in error_images:
        query_out += ('-' * 50 + '\n')
        query_out += (i + '\n')
        for j in error_response[i]:
            query_out += (j + '\n')
    return query_out


def load_model(model_cache_dir, model_type, ckpt_dir):
    temperature = 0.4
    top_p = 0.75
    max_length = 2048

    template_type = get_default_template_type(model_type)
    print(f'template_type: {template_type}')  # template_type: qwen

    model = None
    tokenizer = None
    if 'yi' in model_type:
        model, tokenizer = get_model_tokenizer_yi_vl(model_cache_dir,
                                                     torch_dtype=torch.bfloat16,
                                                     model_kwargs={'device_map': 'auto'})
    else:
        model, tokenizer = get_model_tokenizer(
            model_type=model_type,
            torch_dtype=torch.bfloat16,
            model_id_or_path=model_cache_dir,
            model_kwargs={'device_map': 'auto'})

    if len(ckpt_dir) > 0:
        # 加载lora参数
        ckpt_dir = '/hy-tmp/output/deepseek-vl-7b-chat/v3/checkpoint-6250/'
        # 载入lora参数
        model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)

    template = get_template(template_type, tokenizer)

    seed_everything(42)
    model.generation_config.temperature = temperature
    model.generation_config.max_length = max_length
    model.generation_config.top_p = top_p

    return template, model


# 数据集处理
def get_dataset(val_dataset_path):
    # 加载JSON文件
    df = pd.read_json(val_dataset_path,lines=True if Path(val_dataset_path).suffix == '.jsonl' else False)

    print(df)
    return df



def eval_model(template, model, df, logdir):
    pbar = tqdm(desc='evalating....', total=df.shape[0], leave=None)
    rouger = Rouge()
    img_dict = {}
    error_response = {}
    right_count = 0
    sum_count = 0
    temp_score = {'rouge-1': {'f': 0,
                              'p': 0,
                              'r': 0},
                  'rouge-2': {'f': 0,
                              'p': 0,
                              'r': 0},
                  'rouge-l': {'f': 0,
                              'p': 0,
                              'r': 0}}

    system = '''You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.'''

    for index, row in df.iterrows():
        pbar.update(1)

        query = row['query']
        images = None
        if 'images' in row.keys():
            images = row['images']
        label = row['response']
        if "以下属性" in query:
            continue
        sum_count += 1
        
        record = {}
        # 进行推理
        response = single_infer(template, model, system, query, images, )

        # 计算得分
        label = label.strip()
        response = response.strip()


        scores = rouger.get_scores(' '.join(response), ' '.join(label))

        for r_type in scores[0].keys():
            for every_score in scores[0][r_type].keys():
                temp_score[r_type][every_score] += scores[0][r_type][every_score]


        if label == response:
            right_count += 1




        pbar.set_postfix_str(f'正确的回答有{right_count}个，正确率为{(right_count / sum_count):.4f}, ' +
                             f"rouge-1 F评分为 {temp_score['rouge-1']['f']/sum_count:.4f}, " +
                             f"rouge-2 F评分为 {temp_score['rouge-2']['f']/sum_count:.4f} "
                             )



        with open(logdir + 'log.txt', 'a') as f:
                f.write(f'[query]:    {query}\n')
                f.write(f'[images]:   {images}\n')
                f.write(f'[response]: {response}\n')
                f.write(f'[label]:    {label}\n')
                f.write("="*50+"\n")

    with open(logdir + 'log.txt', 'a') as f:
            f.write(f'正确的回答有{right_count}个，正确率为{(right_count / sum_count):.4f}, ' +
                    f"rouge-1 F评分为 {temp_score['rouge-1']['f']/sum_count:.4f}, " +
                    f"rouge-2 F评分为 {temp_score['rouge-2']['f']/sum_count:.4f} ")


def eval_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache_dir", required=True, type=str, help="Enter a local model path")
    parser.add_argument("--log_dir", type=str, help="Enter a log dir")
    parser.add_argument("--ckpt_path", type=str, help="Enter a checkpoint path")
    parser.add_argument("--dataset_path", required=True, type=str, help="Enter a dataset path")
    parser.add_argument("--model_type", required=True, type=str, help="Enter model type")

    args = parser.parse_args()
    model_cache_dir = args.model_cache_dir
    dataset_path = args.dataset_path
    log_dir=args.log_dir
    
    # 检查modeltype
    model_type = args.model_type

    # 载入日志路径
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    logdir = f'{log_dir}/result/{model_type}/{formatted_datetime}/' if log_dir!=None else f'./result/{model_type}/{formatted_datetime}/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    

    ckpt_path = ""
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path

    template, model = load_model(model_cache_dir=model_cache_dir, model_type=model_type, ckpt_dir=ckpt_path)
    df = get_dataset(dataset_path)

    eval_model(template, model, df, logdir)

    absolute_path = os.path.abspath(logdir)
    print("[INFO:swift] log path: ",absolute_path)

if __name__ == "__main__":
    main()
