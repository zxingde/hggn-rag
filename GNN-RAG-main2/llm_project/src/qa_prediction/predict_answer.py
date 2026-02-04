import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import argparse
import pickle
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial
from src.models.projector import StructureProjector
import torch

import json

with open('entities_names.json') as f:
    entities_names = json.load(f)
names_entities = {v: k for k, v in entities_names.items()}

import re
import string
def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def load_gnn_rag(g_data_file, g_data_file2=None):
    data_file_d = {}
    data_file_gnn = {}

    data_file = os.path.dirname(g_data_file) + "/test.json"
    with open(data_file) as f_in, open(g_data_file) as fg:
        for line, lineg in (zip(f_in, fg)):
            line = json.loads(line)
            lineg = json.loads(lineg)
            
            data_file_d[line["id"]] = line
            data_file_gnn[line["id"]] = lineg
        print("ok1")
    if g_data_file2 is not None:
        data_file = os.path.dirname(g_data_file2) + "/test.json"
        with open(data_file) as f_in, open(g_data_file2) as fg:
            for line, lineg in (zip(f_in, fg)):
                line = json.loads(line)
                lineg = json.loads(lineg)
                
                cand1 = data_file_gnn[line["id"]]["cand"]
                cand2 =  lineg["cand"]

                for c2 in cand2: #c[0] entity c[1] score
                    found=False
                    for c1 in cand1:
                        if c2[0] == c1[0]:
                            if c2[1] > c1[1]: c1[1] = c2[1]
                            found=True
                            break
                    if not found:
                        cand1.append(c2)
                cand1 = sorted(cand1, key=lambda x: x[1], reverse=True)
                data_file_gnn[line["id"]]["cand"] = cand1
            data_file_gnn[line["id"]].update({"cand2": lineg["cand"]})
            print("ok2")

    return data_file_gnn


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset


def prediction(data, processed_list, input_builder, model, encrypt=False, data_file_gnn=None,
               projector=None, graph_features=None, device='cuda'):  # 【修改 1】增加参数
    question = data["question"]
    answer = data["answer"]
    entities = data['q_entity']

    data["cand"] = None
    id = data["id"]
    if data_file_gnn is not None:
        if data["id"] in data_file_gnn:  # 加个安全检查
            lineg = data_file_gnn[data["id"]]
            cand = lineg['cand']
            predictiong = []
            for c in cand:
                if c[0] in entities_names:
                    predictiong.append(entities_names[c[0]])
                else:
                    predictiong.append(c[0])
            data["cand"] = predictiong

    # --- 原有逻辑：跳过已处理 ---
    if id in processed_list:
        return None

    # --- 原有逻辑：无模型模式 (Direct Answer) ---
    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": id,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "input": question,
        }



    # --- 生成文本 Prompt ---
    input_text = input_builder.process_input(data)

    # --- 【修改 2】核心逻辑：准备 Soft Prompts ---
    soft_prompts = None
    # 只有当模型、特征库、投影层都存在时才执行
    if model is not None and graph_features is not None and projector is not None:
        # 使用题目 ID 查找特征
        if id in graph_features:
            try:
                # 1. 获取原始 GNN 特征 (List 或 Tensor)
                gnn_feats = graph_features[id]

                # 2. 确保它是 Tensor
                if not isinstance(gnn_feats, torch.Tensor):
                    gnn_feats = torch.tensor(gnn_feats)

                # 3. 移动到正确的设备 (GPU)
                gnn_feats = gnn_feats.to(device)

                # 4. 增加 Batch 维度: [Nodes, Dim] -> [1, Nodes, Dim]
                # 这一步是为了适应 Projector 的输入要求
                if gnn_feats.dim() == 2:
                    gnn_feats = gnn_feats.unsqueeze(0)

                # 5. 通过投影层生成 Soft Prompts
                # [1, Nodes, 50] -> [1, 2, 4096]
                with torch.no_grad():  # 推理时不需要梯度
                    soft_prompts = projector(gnn_feats)

            except Exception as e:
                print(f"Error processing graph features for {id}: {e}")
                soft_prompts = None

    # --- 【修改 3】模型推理 ---
    # 调用 generate_sentence，传入文本和软提示
    if soft_prompts is not None:
        prediction = model.generate_sentence(input_text, soft_prompts=soft_prompts).strip()
    else:
        # 如果没有特征，就退化为纯文本推理
        prediction = model.generate_sentence(input_text).strip()

    if prediction is None:
        return None
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input_text,
    }
    return result


def main(args, LLM):
    input_file = os.path.join(args.data_path, args.d)
    rule_postfix = "no_rule"
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"

    data_file_gnn = None
    if os.path.exists(args.rule_path_g1):
        if not os.path.exists(args.rule_path_g2):
            data_file_gnn = load_gnn_rag(args.rule_path_g1)
        else: 
            data_file_gnn = load_gnn_rag(args.rule_path_g1, args.rule_path_g2)



    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
        
    print("Load dataset from finished")
    output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, args.split, rule_postfix, str(args.encrypt)
    )
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        input_builder = PromptBuilder(
            args.prompt_path,
            args.encrypt,
            args.add_rule,
            use_true=args.use_true,
            cot=args.cot,
            explain=args.explain,
            use_random=args.use_random,
            each_line=args.each_line,
            maximun_token=model.maximun_token,
            tokenize=model.tokenize,
        )
        print("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        # Directly return last entity as answer
        input_builder = PromptBuilder(
            args.prompt_path, args.encrypt,args.add_rule, use_true=args.use_true
        )
    projector = None
    # 假设 args.device 已经定义 (通常是 'cuda' 或 'cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 这里判断：如果你在命令行传了 --feature_path (说明要用 GNN 特征)，就初始化 projector
    if hasattr(args, 'feature_path') and args.feature_path:
        print(f"Initializing StructureProjector on {device}...")

        # ⚠️ 注意：这里参数要跟你训练时保持一致
        # gnn_dim=50 (你的 GNN 输出维度)
        # llm_dim=4096 (Llama-2-7b 的维度)
        # num_tokens=2 (你设定的固定 Token 数)
        projector = StructureProjector(gnn_dim=50, llm_dim=4096, num_tokens=2)

        # 加载训练好的权重 (如果有)
        if hasattr(args, 'projector_path') and args.projector_path and os.path.exists(args.projector_path):
            print(f"Loading Projector weights from {args.projector_path}")
            projector.load_state_dict(torch.load(args.projector_path, map_location=device))
        else:
            print("⚠️ Warning: Projector using random weights! (Results may be random)")

        projector.to(device)
        # projector.eval()  # 开启推理模式 (冻结 Dropout/Batchnorm 等)
    # ===================================================================
    graph_features = None

    # 定义你的特征文件所在的目录
    feature_dir = "gnn-fet"  # 你说的文件夹名字

    # 自动判断要加载哪个文件
    # args.d 通常包含数据集名字，例如 "RoG-webqsp" 或 "RoG-cwq"
    dataset_name = args.d.lower()
    feature_path = None

    if "cwq" in dataset_name:
        feature_filename = "0117-HGNN-CWQ_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
        feature_path = os.path.join(feature_dir, feature_filename)
        print(f"检测到数据集为 CWQ，准备加载特征文件: {feature_path}")

    elif "webqsp" in dataset_name:
        feature_filename = "0117-HGNN-webqsp_LMSR_BS24_4090_EXPORT_2_graph_features.pkl"
        feature_path = os.path.join(feature_dir, feature_filename)
        print(f"检测到数据集为 WebQSP，准备加载特征文件: {feature_path}")

    else:
        # 如果 args.d 里没写，也可以允许通过参数 --feature_path 强制指定
        if hasattr(args, 'feature_path') and args.feature_path:
            feature_path = args.feature_path
            print(f"未识别数据集，使用命令行指定的特征文件: {feature_path}")
        else:
            print("⚠️ 警告: 未检测到 CWQ 或 WebQSP 关键字，且未指定 feature_path，将跳过 GNN 特征加载。")

    # 执行加载
    if feature_path and os.path.exists(feature_path):
        print(f"正在加载 GNN 特征... 这可能需要几秒钟...")
        try:
            with open(feature_path, 'rb') as f:
                graph_features = pickle.load(f)
            print(f"✅ 成功加载特征库，共包含 {len(graph_features)} 条数据的特征。")

            # 【可选检查】打印一条看看格式对不对
            # first_key = list(graph_features.keys())[0]
            # print(f"示例特征 Shape: {graph_features[first_key].shape}")

        except Exception as e:
            print(f"❌ 加载特征文件失败: {e}")
            graph_features = None
    else:
        if feature_path:
            print(f"❌ 错误: 找不到文件 {feature_path}，请检查路径。")

    # ================================================================
    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                        encrypt=args.encrypt,
                        data_file_gnn=data_file_gnn

                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction(data, processed_list, input_builder, model, encrypt=args.encrypt, data_file_gnn=data_file_gnn,
                             projector=projector,
                             graph_features=graph_features,
                             device=args.device
                             )
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    fout.close()

    eval_result(output_file, encrypt=args.encrypt)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path", type=str, default="rmanluo"
    )
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="gpt-3.5-turbo",
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        help="prompt_path",
        default="prompts/llama2_predict.txt",
    )
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp/RoG/test/predictions_3_False.jsonl",
    )
    argparser.add_argument(
        "--rule_path_g1",
        type=str,
        default="results/gnn/webqsp/RoG/test/rearev-sbert/test.info",
    )
    argparser.add_argument(
        "--rule_path_g2",
        type=str,
        default=None,
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    argparser.add_argument("--encrypt", action="store_true")

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)
