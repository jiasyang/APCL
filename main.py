import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from utils import save, load, train, test, data_to_device, data_concatenate
from datasets_addsentence_pro import MIMIC, IUXRAY,organ_keywords,region_keywords,pixel_keywords
from losses import CELossTotal
from models import CNN, Encoder, TNN, Classifier, Generator, Context



def infer(data_loader, model, device='cuda', threshold=None):
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, (source, target) in enumerate(prog_bar):
            source = data_to_device(source, device)
            target = data_to_device(target, device)
            if threshold != None:
                output = model(image=source[0], history=source[3], threshold=threshold)

            else:
                output = model(source[0])

            outputs.append(data_to_device(output))
            targets.append(data_to_device(target))
        # for i in targets[0][1]:
        #     print(i)

        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)


    return outputs, targets
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
torch.set_num_threads(1)
torch.manual_seed(seed=123)

# Reload = True # True or False  用于是否接着训练/是否推理
# Phase = 'INFER'  # TRAIN or INFER
#
Reload = False # True or False  用于是否接着训练/是否推理
Phase = 'TRAIN'  # TRAIN or INFER

#
DATASET_NAME = 'IUXRAY'
MODEL_NAME = 'Context'


if DATASET_NAME == 'MIMIC':
    EPOCHS = 100
    BATCH_SIZE = 8 if Phase == 'TRAIN' else 64
    MILESTONES = [25]

elif DATASET_NAME == 'IUXRAY':
    EPOCHS = 100
    BATCH_SIZE = 8 if Phase == 'TRAIN' else 64
    MILESTONES = [25]
else:
    raise ValueError('Invalid DATASET')


if __name__ == "__main__":
    if MODEL_NAME in ['Context']:
        SOURCES = ['image','caption','label','history',]
        TARGETS = ['caption','label',]
        KW_SRC = ['image','caption','label','history',]
        KW_TGT = None
        KW_OUT = None
    # if MODEL_NAME in ['Context']:
    #     SOURCES = ['image','caption','label','history',"label_14"]
    #     TARGETS = ['caption','label']
    #     KW_SRC = ['image','caption','label','history',"label_14"]
    #     KW_TGT = None
    #     KW_OUT = None

    else:
        raise ValueError('Invalid BACKBONE_NAME')

    """这里对数据集的选择并设置"""
    if DATASET_NAME == 'MIMIC':
        INPUT_SIZE = (224,224)
        MAX_VIEWS = 2
        NUM_LABELS = 145    ##### 这里是14个标签与100个词汇表中出现的最多的数的标签
        NUM_CLASSES = 2

        dataset = MIMIC('./Context-Enhanced-Framework/mimic_cxr/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=0, generate_splits=False, debug_mode=False)

        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')

    elif DATASET_NAME == 'IUXRAY':
        INPUT_SIZE = (224,224)
        MAX_VIEWS = 2 # 最大视角数   每个描述可能有不同的视角
        NUM_LABELS = 89  # 标签数
        NUM_CLASSES = 2  # 类别数,这里可能指的是有病没病

        dataset = IUXRAY('./Context-Enhanced-Framework/iu_xray/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(seed=123)

        VOCAB_SIZE = len(dataset.vocab) # 词汇表大小
        POSIT_SIZE = dataset.max_len # 词汇表中最大的句子长度
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
        print("conment",COMMENT)
    else:
        raise ValueError('Invalid DATASET_NAME')


    """这里选择了模型名字,因为组合在一起了所以这里只有一个"""
    if MODEL_NAME == 'Context':
        LR = 6e-5  # 设置学习率 之前  6e-5  == 0.00006
        WD = 1e-2 # 权重衰减
        DROPOUT = 0.1 #
        NUM_EMBEDS = 256 # 嵌入维度
        FWD_DIM = 256 # 前馈层维度
        NUM_HEADS = 8 # 注意力头数
        NUM_LAYERS = 2 # 层数
        import pandas as pd
        df = pd.read_csv('RADLEX.csv', sep=',', low_memory=False)


        # 预处理：将目标字段合并为一个统一的文本串，便于搜索关键词
        df['combined'] = (
                df['Preferred Label'].fillna('') + ' ' +
                df['Synonyms'].fillna('') + ' ' +
                df['Definitions'].fillna('') + ' ' +
                df['Semantic Types'].fillna('') + ' ' +
                df['Parents'].fillna('')
        ).str.lower()


        # 分类函数
        def classify_level(text):
            if any(keyword in text for keyword in pixel_keywords):
                return '像素级别 (pixel-level)'
            elif any(keyword in text for keyword in region_keywords):
                return '区域级别 (region-level)'
            elif any(keyword in text for keyword in organ_keywords):
                return '器官级别 (organ-level)'
            return '未知 (unknown)'

        # 应用分类
        df['Level'] = df['combined'].apply(classify_level)

        # 分别提取各级别的 Preferred Label
        organ_terms = df[df['Level'] == '器官级别 (organ-level)']['Preferred Label'].dropna().tolist()
        region_terms = df[df['Level'] == '区域级别 (region-level)']['Preferred Label'].dropna().tolist()
        pixel_terms = df[df['Level'] == '像素级别 (pixel-level)']['Preferred Label'].dropna().tolist()
        print(len(organ_terms)) #14268
        print(len(region_terms)) # 12619
        print(len(pixel_terms))# 1147

        organ_terms=organ_terms[:1000]
        region_terms=region_terms[:1000]
        pixel_terms=pixel_terms[:1000]



        cnn = CNN(pixel_terms,region_terms,organ_terms)  # 定义卷积层 输入其backbone以及使用的backbone
        cnn = Encoder(cnn) # 初始化mvcnn  来提取 avg全局特征以及wxh空间特征
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        NUM_HEADS = 1
        NUM_LAYERS = 12

        cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=1024, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT,data_flag=DATASET_NAME)
        gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)

        model = Context(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
        criterion = CELossTotal(ignore_index=3)
    else:
        raise ValueError('Invalid MODEL_NAME')

    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    print('Total Parameters:', sum(p.numel() for p in model.parameters()))

    last_epoch = -1
    best_metric = 1e9
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    """
    这里还没跑到最好的精度
    """
    checkpoint_path_from = 'checkpoints/{}_{}_{}_rollout.pt'.format(DATASET_NAME,MODEL_NAME,COMMENT)
    checkpoint_path_to = 'checkpoints/{}_{}_{}_rollout.pt'.format(DATASET_NAME,MODEL_NAME,COMMENT)



    if Reload:
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))

    if Phase == 'TRAIN':
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(last_epoch+1, EPOCHS):
            print('Epoch:', epoch)
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, scaler=scaler)
            val_loss = test(val_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            test_loss = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)

            scheduler.step()

            if best_metric > val_loss:
                best_metric = val_loss
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print('New Best Metric: {}'.format(best_metric))
                print('Saved To:', checkpoint_path_to)


    elif Phase == 'INFER':
        print("----------- Infer Beging -----------")
        txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.15)

        gen_outputs = txt_test_outputs[0]
        gen_targets = txt_test_targets[0]
        gen_label = txt_test_targets[1]
        print(gen_label)
        print(len(gen_label))

        print("\n")
        print(gen_outputs)
        print(gen_label.shape)
        print(len(gen_outputs))
        print("\n")
        print(gen_targets)
        print(len(gen_targets))
        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        out_file_ref = open('outputs/x_{}_{}_{}_Ref.txt'.format(DATASET_NAME, MODEL_NAME, COMMENT), 'w')
        out_file_hyp = open('outputs/x_{}_{}_{}_Hyp.txt'.format(DATASET_NAME, MODEL_NAME, COMMENT), 'w')
        out_file_lbl = open('outputs/x_{}_{}_{}_Lbl.txt'.format(DATASET_NAME, MODEL_NAME, COMMENT), 'w')

        tensor_cut = gen_label[:, :14]
        df = pd.DataFrame(tensor_cut.numpy())
        # 保存为 CSV 文件（不带行号和表头）
        df.to_csv("outputs/label.csv", index=False, header=False)


        # 保存到txt文件
        for row in tensor_cut:
            # 把每个值转成字符串并用空格连接
            line = " ".join(map(str, row.tolist()))
            out_file_lbl.write(line + "\n")

        for i in range(len(gen_outputs)): # 遍历每一个生成的文本
            candidate = ''  # 这里表示生成的文本
            for j in range(len(gen_outputs[i])): # 对每一个文本数据遍历,得到词语
                tok = dataset.vocab.id_to_piece(int(gen_outputs[i,j])) # 将词汇id转换为文本片段
                if tok == '</s>': # 遇到结束符号就截至
                    break
                elif tok == '<s>': # 代表着开始
                    continue
                elif tok == '▁': # 这里表示空格
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' '# 如果当前生成的文本不为空,且最后一个字符不是空格,则添加一个空格
                elif tok in [',', '.', '-', ':']:
                    if len(candidate) and candidate[-1] != ' ': # 在标点符号前后添加空格
                        candidate += ' ' + tok + ' '
                    else:
                        candidate += tok + ' '
                else:
                    candidate += tok
            out_file_hyp.write(candidate + '\n')  # 将生成的文本写入文件

            reference = '' # 这是参考文本目标标签
            for j in range(len(gen_targets[i])):
                tok = dataset.vocab.id_to_piece(int(gen_targets[i,j]))
                if tok == '</s>':
                    break
                elif tok == '<s>':
                    continue
                elif tok == '▁':
                    if len(reference) and reference[-1] != ' ':
                        reference += ' '
                elif tok in [',', '.', '-', ':']:
                    if len(reference) and reference[-1] != ' ':
                        reference += ' ' + tok + ' '
                    else:
                        reference += tok + ' '
                else: # letter
                    reference += tok
            out_file_ref.write(reference + '\n')

        target_label=txt_test_targets[1]
        generate_label=txt_test_outputs[1]
        print(target_label)   ### 这里使用得是
        generate_probs = generate_label[:, :, 1]  # 取正类概率，形状 (4, 25)
        generate_binary = (generate_probs > 0.5).float()  # 阈值0.5二值化
        print(generate_binary)


    else:
        raise ValueError('Invalid Phase')


