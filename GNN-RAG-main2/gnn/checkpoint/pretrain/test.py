import pickle
with open('0117-HGNN-WebQSP_LMSR_BS24_4090_EXPORT_all_features.pkl', 'rb') as f:
    data = pickle.load(f)
    print(len(data)) # 检查数量
    print(next(iter(data.values())).shape) # 检查单个特征维度