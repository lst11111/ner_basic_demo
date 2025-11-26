from huggingface_hub import snapshot_download

##下载模型权重
model_dir = snapshot_download(
                            repo_id="hfl/chinese-bert-wwm",
                            local_dir="/home/lisongtao/ner_basic_demo/pre_model/chinese-bert-wwm",
                            ignore_patterns=["*.h5", "*.ot", "tf_model*"] )
print(f"模型已经下载到: {model_dir}")
