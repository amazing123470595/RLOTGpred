import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
from tqdm import tqdm

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", local_files_only=False)

device = torch.device('cpu')
model = model.to(device)
model = model.eval()


def find_features_full_seq(sequence):
    sequence = re.sub(r"[UZOB]", "X", sequence)
    # 提取字母K的位置
    sequence = ' '.join(sequence)
    ids = tokenizer.encode_plus(sequence, add_special_tokens=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        embedding = model(**ids)[0]
    embedding = embedding.squeeze(0).cpu().numpy()
    seq_len = (ids['attention_mask'][0] == 1).sum().item()
    seq_emd = embedding[:seq_len - 1]
    return seq_emd


def process_sequences_file(input_file, output_file, start_index=0):
    batch_size = 8
    with open(input_file, 'r') as infile, open(output_file, 'a') as outfile:
        sequences_batch = []
        total_sequences = sum(1 for line in infile) // 2
        infile.seek(0)
        for _ in range(start_index):
            next(infile)
            next(infile)
        with tqdm(total=total_sequences - start_index) as pbar:
            while True:
                title = infile.readline().strip()
                if not title:
                    break
                sequence = infile.readline().strip()
                sequences_batch.append((title, sequence))

                if len(sequences_batch) == batch_size:
                    features_batch = [find_features_full_seq(seq[1]) for seq in sequences_batch]
                    for title_seq, features in zip(sequences_batch, features_batch):
                        outfile.write(title_seq[0] + ',')  # 写入标题
                        outfile.write(','.join(map(str, features.flatten())) + '\n')  # 写入内容
                    sequences_batch = []
                    pbar.update(batch_size)

        if len(sequences_batch) > 0:
            features_batch = [find_features_full_seq(seq[1]) for seq in sequences_batch]
            for title_seq, features in zip(sequences_batch, features_batch):
                outfile.write(title_seq[0] + ',')  # 写入标题
                outfile.write(','.join(map(str, features.flatten())) + '\n')  # 写入内容
            pbar.update(len(sequences_batch))


N_input_file_path = 'MyData/neg_Independent.fasta'
N_output_file_path = 'MyData/neg_Independent_CD_ProtT5_features.csv'

P_input_file_path = 'MyData/pos_Independent.fasta'
P_output_file_path = 'MyData/pos_Independent_CD_ProtT5_features.csv'

# 指定从哪个序列开始处理（如果是首次运行，则从头开始）
start_index = 0  # 修改为已处理的序列数量

# 提取特征并写入文件，并显示进度
process_sequences_file(N_input_file_path, N_output_file_path, start_index)
process_sequences_file(P_input_file_path, P_output_file_path, start_index)
