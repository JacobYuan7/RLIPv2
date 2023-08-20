# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools
from random import choice, uniform
from transformers import RobertaModel, RobertaTokenizerFast, BertTokenizerFast, BertModel, AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import json
from collections import OrderedDict
import torch
from typing import List
import pdb
# pdb.set_trace()


def extract_textual_features_automodel(text_encoder_type = "princeton-nlp/sup-simcse-roberta-large"):
    '''
    text_encoder_type: example: ["princeton-nlp/sup-simcse-bert-base-uncased", "princeton-nlp/sup-simcse-roberta-large", ...]
    '''
    # Import our models. The package will take care of downloading the models automatically
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    model = AutoModel.from_pretrained(text_encoder_type)
    
    # Tokenize input texts
    with open("/mnt/data-nas/peizhi/jacob/OCN/datasets/vg_keep_names_v1_no_lias_freq.json", "r") as f:
        vg_keep_names = json.load(f)
    relationship_names = vg_keep_names["relationship_names"]
    object_names = vg_keep_names["object_names"]
    print(f'Loading json finished, with {len(relationship_names)} relationships and {len(object_names)} objects loaded.')
    len_rel, len_obj = len(relationship_names), len(object_names)
    len_text, flat_text = len_rel + len_obj, relationship_names + object_names

    text_feature = []
    flag = 0
    while flag < len_text:
        if flag + 1000 < len_text:
            partial_text = flat_text[flag: (flag+1000)]
            flag += 1000
        else:
            partial_text = flat_text[flag:]
            flag = len_text 
        print(flag)
        
        # Get the embeddings
        inputs = tokenizer(partial_text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        text_feature.append(embeddings.detach())
    text_feature = torch.cat(text_feature, dim = 0)
    print('Feature extraction finished!')

    rel_feature = {}
    obj_feature = {}
    for idx, f in enumerate(text_feature):
        if idx < len_rel:
            rel_feature[flat_text[idx]] = f.numpy()
        else:
            obj_feature[flat_text[idx]] = f.numpy()
    print(len(rel_feature), len(obj_feature))
    np.savez_compressed(f"/mnt/data-nas/peizhi/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_{text_encoder_type.split('/')[1]}.npz",
                        rel_feature = rel_feature,
                        obj_feature = obj_feature)
    

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    # cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
    # cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
    # print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
    # print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))


def extract_textual_features_from_mymodel(text_encoder_type = "roberta-base"):
    ### model and tokenizer from checkpoints
    checkpoint = torch.load('/mnt/data-nas/peizhi/logs/ParSe_VG_BiasT_Freq500_GIoUVb_PseuEuc03_aftNorm_COCO_bs128_150e/checkpoint0149.pth')
    model = checkpoint['model']
    model_text_encoder = {}
    for i, j in model.items():
        if 'transformer.text_encoder.' in i:
            model_text_encoder[i.replace('transformer.text_encoder.', '')] = j
    tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
    text_encoder = RobertaModel.from_pretrained(text_encoder_type, state_dict = model_text_encoder)
    
    ### model and tokenizer from Roberta
    # tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
    # text_encoder = RobertaModel.from_pretrained(text_encoder_type)

    with open("/mnt/data-nas/peizhi/jacob/OCN/datasets/vg_keep_names_v1_no_lias_freq.json", "r") as f:
        vg_keep_names = json.load(f)
    print('Loading json finished.')
    relationship_names = vg_keep_names["relationship_names"]
    # print(relationship_names)
    object_names = vg_keep_names["object_names"]
    print(len(relationship_names), len(object_names))
    if "relationship_freq" in vg_keep_names.keys():
        relationship_freq = vg_keep_names["relationship_freq"]
    if "object_freq" in vg_keep_names.keys():
        object_freq = vg_keep_names["object_freq"]
    
    len_rel = len(relationship_names)
    len_obj = len(object_names)
    len_text = len_rel + len_obj
    flat_text = relationship_names + object_names
    # flat_text = flat_text[-1001:]
    # len_text = 1001
    
    text_feature = []
    flag = 0
    while flag < len_text:
        if flag + 1000 < len_text:
            partial_text = flat_text[flag: (flag+1000)]
            flag += 1000
        else:
            partial_text = flat_text[flag:]
            flag = len_text 
        print(flag)

        flat_tokenized = tokenizer.batch_encode_plus(partial_text, padding="longest", return_tensors="pt")
        # tokenizer: dict_keys(['input_ids', 'attention_mask'])
        #            'input_ids' shape: [text_num, max_token_num]
        #            'attention_mask' shape: [text_num, max_token_num]
        encoded_flat_text = text_encoder(**flat_tokenized)
        text_memory = encoded_flat_text.pooler_output
        text_feature.append(text_memory.detach())
        # text_feature.update({t:m for t, m in zip(partial_text, text_memory)})
    text_feature = torch.cat(text_feature, dim = 0)
    print('Feature extraction finished!')
    # m = text_feature[[0,8,10]]
    # print(torch.einsum('ab,cb->ac',m,m))

    rel_feature = {}
    obj_feature = {}
    for idx, f in enumerate(text_feature):
        if idx < len_rel:
            rel_feature[flat_text[idx]] = f.numpy()
        else:
            obj_feature[flat_text[idx]] = f.numpy()
    print(len(rel_feature), len(obj_feature))
    
    np.savez_compressed('/mnt/data-nas/peizhi/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_RLIP-ParSe_COCO_VG.npz',
                        rel_feature = rel_feature,
                        obj_feature = obj_feature)
    


    # test similarity
    # print(rel_feature['on'].T)
    # test_text = [rel_feature['on'], rel_feature['on a'], rel_feature['on top of'], , rel_feature['are on']]

    # text_memory_resized = model.module.transformer.resizer(text_memory)
    # text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, args.batch_size, 1)
    # # text_attention_mask = torch.ones(text_memory_resized.shape[:2], device = device).bool()
    # text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
    # text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)
    # # kwargs = {'text':text}

def extract_hico_verb_features(text_encoder_type = "roberta-base"):
    # checkpoint = torch.load('/mnt/data-nas/peizhi/logs/ParSe_VG_BiasT_Freq500_GIoUVb_PseuEuc03_aftNorm_COCO_bs128_150e/checkpoint0149.pth')
    # model = checkpoint['model']
    # model_text_encoder = {}
    # for i, j in model.items():
    #     if 'transformer.text_encoder.' in i:
    #         model_text_encoder[i.replace('transformer.text_encoder.', '')] = j
    # tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
    # text_encoder = RobertaModel.from_pretrained(text_encoder_type, state_dict = model_text_encoder)
    
    tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
    text_encoder = RobertaModel.from_pretrained(text_encoder_type)

    hico_verb_list = load_hico_verb_txt()
    flat_tokenized = tokenizer.batch_encode_plus(hico_verb_list, padding="longest", return_tensors="pt")
    # tokenizer: dict_keys(['input_ids', 'attention_mask'])
    #            'input_ids' shape: [text_num, max_token_num]
    #            'attention_mask' shape: [text_num, max_token_num]
    encoded_flat_text = text_encoder(**flat_tokenized)
    text_memory = encoded_flat_text.pooler_output
    text_feature = text_memory.detach()

    print('Feature extraction finished!')

    hico_verb_dict = {v:f for v, f in zip(hico_verb_list, text_feature)}
    # np.savez_compressed('/mnt/data-nas/peizhi/data/hico_20160224_det/word_embedding/hico_verb_RLIP-ParSe_COCO_VG.npz',
    #                     hico_verb_dict = hico_verb_dict)
    np.savez_compressed('/mnt/data-nas/peizhi/data/hico_20160224_det/word_embedding/hico_verb_Original_RoBERTa.npz',
                        hico_verb_dict = hico_verb_dict)


def load_hico_verb_txt(file_path = '/mnt/data-nas/peizhi/jacob/OCN/datasets/hico_verb_names.txt') -> List[list]:
    '''
    Output like [['train'], ['boat'], ['traffic', 'light'], ['fire', 'hydrant']]
    '''
    verb_names = []
    for line in open(file_path,'r'):
        # verb_names.append(line.strip().split(' ')[-1])
        verb_names.append(' '.join(line.strip().split(' ')[-1].split('_')))
    return verb_names


def check_similar_words(anchor_word = 'man'):
    ### You can refer to an example from huggingface: https://github.com/princeton-nlp/SimCSE#use-simcse-with-huggingface
    # obj_feature = np.load('/mnt/data-nas/peizhi/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_text_feature.npz', allow_pickle = True)['obj_feature'].item()
    # obj_feature = np.load('/mnt/data-nas/peizhi/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_sup-simcse-roberta-large.npz', allow_pickle = True)['obj_feature'].item()
    obj_feature = np.load('/mnt/data-nas/peizhi/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_sup-simcse-roberta-base.npz', allow_pickle = True)['obj_feature'].item()
    obj_feature = {i:torch.from_numpy(j) for i,j in obj_feature.items()}  

    with open("/mnt/data-nas/peizhi/jacob/RLIP/datasets/vg_keep_names_v1_no_lias_freq.json", "r") as f:
        vg_keep_names = json.load(f)
    relationship_names = vg_keep_names["relationship_names"]
    object_names = vg_keep_names["object_names"]
    print(f'Loading json finished, with {len(relationship_names)} relationships and {len(object_names)} objects loaded.')

    if anchor_word in object_names:
        anchor_feature = obj_feature[anchor_word]
        sim_dict = OrderedDict()
        # for i,j in items():
        #     sim_dict[i] = 1 - cosine(anchor_feature, j)
        sim_dict = {i:1 - cosine(anchor_feature, j) for i,j in obj_feature.items()}
        sorted_sim = list(sorted(sim_dict.items(), key = lambda x:x[1], reverse = True))
        print(sorted_sim[:100])
    
    ### vg_keep_names_v1_no_lias_freq_sup-simcse-roberta-large.npz
    # anchor_word: "man"
    # sim_dict: [('man', 1.0), ('man.', 0.987231433391571), ('man man', 0.9860828518867493), ('man .', 0.9659639596939087), ('man is', 0.957766056060791), ("man's", 0.9551363587379456), ('is a man', 0.9510306715965271), ('man has', 0.9462917447090149), ('there is a man', 0.9403985738754272), ('man in', 0.9320536255836487), ('man\\', 0.9294561147689819), ('man`', 0.9249423146247864), ('man (he)', 0.9220535159111023), ('this is a man', 0.9215567708015442), ('man (his)', 0.9207451939582825), ('male human', 0.9184167981147766), ('man is visible', 0.9162415266036987), ('human male', 0.9083154201507568), ('man at', 0.9056752324104309), ('this man', 0.8980408310890198), ('man with', 0.8966150283813477), ('man figure', 0.8745579719543457), ('man is wearing', 0.8743029236793518), ('man can', 0.8637485504150391), ('on man', 0.8623335361480713), ('man in clothes', 0.8622342944145203), ("man's picture", 0.8621859550476074), ('picture of a man', 0.8588035702705383), ('photo of a man', 0.8558799028396606), ('near the man', 0.8550971746444702), ('man wearing a', 0.8547655344009399), ('man in area', 0.8534818291664124), (',man', 0.846731960773468), ('man wears a', 0.845801830291748), ('man wearing', 0.8254711627960205), ('reflex of a man', 0.8228245377540588), ('guy', 0.8185068964958191), ('man/outside', 0.8154369592666626), ('another man', 0.8153104782104492), ('caucasian man', 0.8150848150253296), ('he (man)', 0.8137494325637817), ('man in background', 0.8137195110321045), ('male', 0.8117749691009521), ('he face of a man', 0.8044268488883972), ('man overooking', 0.7988713383674622), ("man's clothes", 0.7975930571556091), ('man leftear', 0.7959189414978027), ('face of man', 0.7957667708396912), ("man'", 0.7952361106872559), ('man in shirt', 0.7923057675361633), ('face of a man', 0.7900975942611694), ("man's clothing", 0.7881795167922974), ('man/shirt', 0.786787211894989), ('man in a shirt', 0.7859592437744141), ('man 2', 0.7835241556167603), ("man's body", 0.783317506313324), ('man in front', 0.7796104550361633), ('man with shirt', 0.7770967483520508), ('man has hand', 0.7748644351959229), ('man outdoors', 0.7747936248779297), ('hands of man', 0.7743378281593323), ('man face', 0.7741992473602295), ('man arms', 0.7741932272911072), ('man 1', 0.7740939259529114), ("man's face", 0.7737535238265991), ('man dressed', 0.7737250924110413), ('hand of a man', 0.7721660733222961), ('man has mouth', 0.7703776359558105), ('man has two legs', 0.7686744332313538), ('man shuolder', 0.7682600617408752), ('guy wearing', 0.7667478322982788), ('.man', 0.7622535824775696), ('hand of the man', 0.7618181705474854), ("man's hands", 0.7602953910827637), ('one man', 0.7568016648292542), ('man0s feet', 0.755813479423523), ('man`s face', 0.7555667757987976), ('clothes man', 0.7532649040222168), ('ear of a man', 0.7520214319229126), ('man hand', 0.7505601048469543), ('man with arm', 0.749523401260376), ('man silhouette', 0.7493039965629578), ('man wearing a shirt', 0.7484111189842224), ('manhand', 0.7472260594367981), ('man hands', 0.7471755743026733), ("man's side", 0.7470476627349854), ("man';s hands", 0.7464983463287354), ('man arm', 0.7464284896850586), ('image man', 0.7446902990341187), ('man wearing shirt', 0.7435048818588257), ("man's hand", 0.7425016760826111), ("man's profile", 0.7421046495437622), ('man on the right', 0.739285409450531), ('man mouth', 0.7374557256698608), ('man and', 0.7371577620506287), ("man's appron", 0.73589688539505), ('mouth of a man', 0.735683023929596), ("man's legs", 0.7353058457374573), ('man outfit', 0.7351990342140198), ('man has hair', 0.7350057363510132)]
    # anchor_word: "person"
    # sim_dict: [('person', 1.0), ('person person', 0.9719043970108032), ('person.', 0.9704770445823669), ('person  is', 0.9409026503562927), ('person\\', 0.9295095801353455), ("person's", 0.9272814393043518), ('human', 0.9216678738594055), ('someone', 0.9171213507652283), ('human being', 0.9169129729270935), ('that is a person', 0.9093868732452393), ('individual', 0.9059092402458191), ('this person is', 0.9055731296539307), ('person w', 0.90132737159729), ("someone's", 0.9002405405044556), ('this is a person', 0.8868348002433777), ('person is', 0.8836377859115601), ('this person', 0.8756253123283386), ('personleg', 0.8663191795349121), ('person figure', 0.8635802268981934), ('person at', 0.860186755657196), ('person in  picture', 0.8584557175636292), ('person in', 0.8561344146728516), ('human figure', 0.8548992872238159), ('another person', 0.8532748222351074), ('person in picture', 0.8494771122932434), ("person's ar,", 0.847411036491394), ('person with', 0.8467070460319519), ("person's body", 0.8465882539749146), ('person picture', 0.841969907283783), ("person's image", 0.8362089991569519), ('person in the photo', 0.8354632258415222), ('personhand', 0.8331624269485474), ('person i', 0.8312105536460876), ('ear of a person', 0.8284873962402344), ("person's shape", 0.8239548206329346), ('pic of person', 0.8237716555595398), ('part of a person', 0.8229262828826904), ('person is wearing', 0.8197370171546936), ("person'", 0.8177208304405212), ('person image', 0.8156379461288452), ('body of a person', 0.809908926486969), ('person wearing', 0.8061209917068481), ('human body', 0.8046786189079285), ("person's face", 0.7993273138999939), ('somebody', 0.7987059950828552), ('person shape', 0.7939481735229492), ('person using', 0.7872798442840576), ('"person', 0.7861477136611938), ('face of a person', 0.7859145402908325), ('person doing', 0.7850121855735779), ('person face', 0.7814263701438904), ('person 2', 0.7798214554786682), ('an', 0.7779909372329712), ('humanoid figure', 0.7773166298866272), ('pic human', 0.7736126184463501), ('person photo', 0.7718225717544556), ('distant person', 0.7700187563896179), ("person's silhouette", 0.7692897915840149), ('human face', 0.7616404294967651), ('human image', 0.760343074798584), ('manear', 0.7593114972114563), ('front of a person', 0.7564191222190857), ('person dressed', 0.7554585933685303), ('to a person', 0.7554304003715515), ('hand of person', 0.7516777515411377), ('hand of a person', 0.7509390711784363), ('persona', 0.7495648264884949), ("person's hand2", 0.7490348815917969), ('hands of person', 0.7487451434135437), ('person/shirt', 0.7468218207359314), ('personhead', 0.7393121123313904), ('person silhouette', 0.7392504811286926), ('person made', 0.73916095495224), ('person to the left', 0.7366101145744324), ('image person', 0.7347920536994934), ('person hand', 0.7338025569915771), ('head of a person', 0.7337188720703125), ('legs person', 0.732931911945343), ('torso of a person', 0.7280920147895813), ('persons head', 0.7242336273193359), ('person ground', 0.7238698601722717), ('leg/person', 0.7226918935775757), ('in the picture', 0.7226356267929077), ("person's head", 0.7224372029304504), ('someon', 0.7223803997039795), ('mouth of a person', 0.7221623659133911), ('person on side', 0.7221596240997314), ('persons leg', 0.7214283347129822), ('personel', 0.7210524082183838), ('person outline', 0.7201088070869446), ('person c', 0.7188277840614319), ('hand/person', 0.7184202075004578), ('person shoulder', 0.7178424000740051), ('hand person', 0.7166292667388916), ('person head', 0.7160197496414185), ("person's hand", 0.713657557964325), ('feet person', 0.7122235298156738), ('sttwo person', 0.7107999920845032), ('person arm', 0.710170567035675), ('feet of the person', 0.7095651030540466)]
    # anchor_word: "people"
    # sim_dict: [('people', 1.0), ('people.', 0.9862441420555115), ('peoples', 0.9758768081665039), ('peoples.', 0.9651427268981934), ("people's", 0.9646188616752625), ('people\\', 0.9531923532485962), ('humans', 0.9527957439422607), ('persons', 0.9505757689476013), ('people have', 0.94949871301651), ('some people', 0.9435691833496094), ('peopl', 0.9409072399139404), ('individuals', 0.9405298829078674), ('several people', 0.9248822331428528), ('other people', 0.9246737957000732), ('people belonging', 0.9218301773071289), ('human beings', 0.9214470386505127), ('tpeople', 0.921056866645813), ('some peopel', 0.9181718230247498), ('there are people', 0.9171413779258728), ('multiple people', 0.910781979560852), ('people around', 0.9075663685798645), ('peopel', 0.9051288962364197), ('on people', 0.9020569920539856), ('wpeople', 0.901142954826355), ('persong', 0.9000964760780334), ('peoople', 0.899796187877655), ('these people', 0.8975376486778259), ('variety of people', 0.8934916853904724), ('groupof people', 0.8859005570411682), ('people in', 0.8800919055938721), ('bunch of people', 0.8778098225593567), ('groups of people', 0.8775085210800171), ('peolple', 0.8761570453643799), ('people in picture', 0.8752190470695496), ('group people', 0.8713914155960083), ('collection of people', 0.8684386014938354), ('group of people', 0.866550087928772), ('group of people.', 0.8641633987426758), ('men and women', 0.8637980818748474), ('people group', 0.8631406426429749), ('suffolk', 0.861414909362793), ('personslap', 0.8613090515136719), ('people in the photo', 0.8501288890838623), ('people are wearing', 0.8470645546913147), ('other people2', 0.8469803333282471), ('people grup', 0.8353596925735474), ('people in background', 0.8352895975112915), ('people out', 0.8309844732284546), ('pictures of people', 0.8303174376487732), ('peoplegroup', 0.8257279992103577), ('persons in front', 0.8251725435256958), ('people are on', 0.8231685161590576), ('foreground people', 0.8183071613311768), ('poeple', 0.8168101906776428), ('in a group', 0.8162975311279297), ('group/people', 0.8092824220657349), ('companions', 0.8051154613494873), ('people photo', 0.804479718208313), ('more people', 0.8036952018737793), ('group of two people', 0.8021963834762573), ('men/women', 0.7978060841560364), ('other people1', 0.7954444885253906), ('people behind', 0.7935125231742859), ('are close together', 0.7903572916984558), ('civilians', 0.7870521545410156), ('many people', 0.7862697243690491), ('peope', 0.7860576510429382), ('folk', 0.7838500738143921), ('people are gathering', 0.7811112999916077), ('people dressed', 0.7803743481636047), ('passenngers', 0.780295729637146), ('adults and children', 0.779543399810791), ('peop;e', 0.7789917588233948), ('people gathering', 0.7777297496795654), ('persons back', 0.7776394486427307), ('crowd people', 0.7749640941619873), ('lots of people', 0.7743058204650879), ('crowd/people', 0.7732511162757874), ('cumpls', 0.771397590637207), ('people on the side', 0.7703629732131958), ('people in distance', 0.7676624655723572), ('bodys', 0.7615719437599182), ('blurry people', 0.7559496164321899), ('"people"', 0.755948007106781), ('background people', 0.7553142309188843), ('couple of people', 0.7528793811798096), ('crowd of people', 0.7517161965370178), ('group of', 0.748310923576355), ('couples', 0.7480754852294922), ('people are enjoying', 0.7474316358566284), ('couple people', 0.7442169785499573), ('people outside', 0.7431913614273071), ('people enjoying', 0.7429892420768738), ('persoin', 0.7417018413543701), ('some adults', 0.7405657768249512), ('people character', 0.7403987646102905), ('crowd part', 0.740138053894043), ('group gathered', 0.7333914041519165), ("people's head", 0.7327054142951965), ('persn', 0.7321640849113464)]

    # anchor_word: "man"
    # [('man', 1.0), ('man man', 0.9772267937660217), ('is a man', 0.9519599676132202), ('man is', 0.9514556527137756), ('man.', 0.9473358988761902), ('there is a man', 0.9386788010597229), ('man has', 0.9375312924385071), ('this is a man', 0.9326640963554382), ('this man', 0.9230893850326538), ('man with', 0.9127544164657593), ('man .', 0.9125559329986572), ('man in', 0.9068413972854614), ('man (his)', 0.8954777121543884), ('man is visible', 0.8945953249931335), ('male human', 0.8942216038703918), ('human male', 0.8919301629066467), ('near the man', 0.8914737105369568), ('man at', 0.8851882219314575), ('man is wearing', 0.8835185170173645), ("man's", 0.8834593296051025), ('man in area', 0.8722564578056335), ('man (he)', 0.8669701218605042), ('man weariing', 0.8646854758262634), ('man\\', 0.8631190657615662), ('man in background', 0.8590893745422363), ('man and', 0.8553505539894104), ('male', 0.8451536297798157), ('he (man)', 0.8427845239639282), ('.man', 0.8426216244697571), ("man'", 0.8394646644592285), ("man's appron", 0.8373826146125793), ('man wears a', 0.8371134996414185), ('on man', 0.8305020332336426), ('arm of a man', 0.8293573260307312), ('man`', 0.8284174203872681), ("man's body", 0.8278408050537109), ("man's side", 0.8227373957633972), ('man1', 0.818370521068573), ('hand of a man', 0.813123345375061), (',man', 0.8128811717033386), ('hand of the man', 0.8115816116333008), ('man wearing', 0.8097426295280457), ('guy', 0.8073040246963501), ('manb', 0.8068836331367493), ('arm of man', 0.8050886988639832), ('reflex of a man', 0.8038849830627441), ('man/outside', 0.8013871312141418), ('another man', 0.8011882901191711), ('man wearing a', 0.7959111928939819), ('man outdoors', 0.7946016788482666), ('other man', 0.7919303178787231), ('man in front', 0.7892661094665527), ('ear of a man', 0.7887683510780334), ('manp', 0.7864295244216919), ('hooded man', 0.7861301898956299), ('"man', 0.7844605445861816), ('man mustace', 0.7834160923957825), ('he is', 0.7832679748535156), ('hand of man', 0.7776153087615967), ('this guy', 0.7765370607376099), ('hands of man', 0.7760007381439209), ('head of a man', 0.7722393274307251), ('man on the right', 0.7715921401977539), ('mman', 0.7715737223625183), ('man w/hat', 0.771031379699707), ('man on a', 0.7704445123672485), ('shorthair man', 0.768994152545929), ('man shuolder', 0.7689267992973328), ('man figure', 0.7677424550056458), ('hat man', 0.7675667405128479), ('jeans man', 0.7652349472045898), ('man in clothes', 0.7646327018737793), ('head of the man', 0.7642232775688171), ('picture of a man', 0.7606472373008728), ('man`s back', 0.7603075504302979), ('leg of a man', 0.7577346563339233), ('thumb of the man', 0.7552112340927124), ('man has mouth', 0.754004180431366), ('ropes the man', 0.7539010643959045), ('fa man', 0.7536349892616272), ('man w/shorthair', 0.7536231875419617), ('man in shirt', 0.7522099018096924), ("man's back", 0.7505729794502258), ('sitted man', 0.7504477500915527), ('back of a man', 0.7500923275947571), ("man's legs", 0.7479644417762756), ('man can', 0.7442255616188049), ('foot of a man', 0.7437325716018677), ('he leg of a man', 0.7436144351959229), ('working man', 0.7428798675537109), ('man`s legs', 0.7418757677078247), ('man w/longhair', 0.7417874336242676), ('man has hand', 0.7414042949676514), ('hair of a man', 0.7402675151824951), ('man outfit', 0.7401509881019592), ('man shirt', 0.7390747666358948), ('man w/cap', 0.7390687465667725), ('man in a shirt', 0.7386543154716492), ('man has hair', 0.7385661005973816), ('one man', 0.738409698009491)]
    # anchor_word: "person"
    # [('person', 1.0), ('person person', 0.9726448059082031), ('person  is', 0.9605469107627869), ('individual', 0.9520815014839172), ('person.', 0.9511179327964783), ('persona', 0.940105140209198), ('human', 0.9367017149925232), ('someone', 0.9356125593185425), ('this is a person', 0.9305621981620789), ("person'", 0.926182210445404), ('this person', 0.9244335889816284), ('person is', 0.923191249370575), ('this person is', 0.9223414659500122), ('person w', 0.9201169013977051), ('person\\', 0.917487621307373), ('human being', 0.9130513668060303), ("person's", 0.9127455949783325), ('personel', 0.9017550349235535), ('person with', 0.8997018933296204), ('ear person', 0.8978763222694397), ("someone's", 0.8912742137908936), ('another person', 0.8909275531768799), ('part of a person', 0.8909174203872681), ('body of a person', 0.8900958895683289), ('person in the photo', 0.8891295790672302), ('person in', 0.8888692855834961), ('person in  picture', 0.8840555548667908), ('person at', 0.8818547129631042), ('person in picture', 0.8762214183807373), ('personleg', 0.8747938275337219), ('there is a', 0.8725505471229553), ('ear of a person', 0.8707736730575562), ('person c', 0.8682110905647278), ('person background', 0.8680440783500671), ("person's body", 0.864870011806488), ('human figure', 0.8624135255813599), ('personhead', 0.8576942086219788), ("person's ar,", 0.8548334836959839), ('person figure', 0.8543370366096497), ('"person', 0.8535057306289673), ('person is wearing', 0.8486717343330383), ('citizen', 0.8465894460678101), ('arm of a person', 0.8455981612205505), ('personthigh', 0.8454494476318359), ('this is one person', 0.8392136096954346), ('jeans person', 0.8387612700462341), ('that is a person', 0.8379887342453003), ('background person', 0.8363324999809265), ("person's shape", 0.8340080380439758), ('bloke', 0.8334790468215942), ('head of a person', 0.8309652805328369), ('reflection of person', 0.8309157490730286), ('humanoid', 0.8308213353157043), ('person image', 0.830268144607544), ('person i', 0.8269250392913818), ('front of a person', 0.8264625072479248), ('torso of a person', 0.8230156898498535), ('hand of a person', 0.8227125406265259), ('hand of person', 0.8189741373062134), ('pic human', 0.8177385330200195), ('person/jeans', 0.815984845161438), ('pic of person', 0.8136643171310425), ('in the foreground', 0.8136558532714844), ('person on side', 0.8136502504348755), ('hands of person', 0.8107892870903015), ('bystander', 0.8097472190856934), ('foloer', 0.8071389198303223), ('person wearing', 0.806618332862854), ('person in distance', 0.8061222434043884), ('personhand', 0.8032907843589783), ('mouth of a person', 0.8027553558349609), ('in the picture', 0.8014522790908813), ("person's legs/feet", 0.7982795834541321), ('person top', 0.7969363927841187), ('somebody', 0.7966359853744507), ('person shape', 0.7941142320632935), ('person doing', 0.7934885025024414), ('humanoid figure', 0.7931061387062073), ('person icon', 0.7924872040748596), ('one person', 0.7920621633529663), ('hat on a person', 0.7908272743225098), ("person's back", 0.7895306348800659), ('feet of the person', 0.7891365885734558), ('an', 0.7890492677688599), ('nose of a person', 0.7881215810775757), ('leg of a person', 0.786286473274231), ("person's eye", 0.7855711579322815), ("person's face", 0.7853671312332153), ('feet of a person', 0.7849189639091492), ("person's head", 0.7835343480110168), ('shadow of the person', 0.7829744815826416), ('shadow of person', 0.7825794219970703), ('person picture', 0.7822227478027344), ("person's image", 0.7808263301849365), ('human body', 0.780799925327301), ('person/bench', 0.7802990078926086), ('person outline', 0.7783700823783875), ('person outside', 0.7782111167907715), ('skeleton of person', 0.7781505584716797), ('is wearing', 0.7766648530960083)]
    # anchor_word: "people"
    # 

    ### vg_keep_names_v1_no_lias_freq_text_feature.npz
    # anchor_word: "man"
    # sim_dict: [('man', 1.0), ('men', 0.9998029470443726), ('king', 0.9997260570526123), ('woman', 0.9997073411941528), ('dog', 0.9996993541717529), ('guy', 0.9996832609176636), ('boy', 0.9996805191040039), ('ana', 0.9996771812438965), ('john', 0.9996739625930786), ('kid', 0.9996735453605652), ('crew', 0.9996702671051025), ('mens', 0.9996693134307861), ('mar', 0.9996682405471802), ('hunter', 0.9996645450592041), ('human', 0.9996613264083862), ('kin', 0.9996609687805176), ('bedroom', 0.9996596574783325), ('ama', 0.9996594786643982), ('women', 0.9996582269668579), ('rooms', 0.9996565580368042), ('buck', 0.9996539950370789), ('standing', 0.9996534585952759), ('grown', 0.9996533989906311), ('moon', 0.9996511340141296), ('hip', 0.9996499419212341), ('ma', 0.9996495842933655), ('animal', 0.9996491074562073), ('mist', 0.9996462464332581), ('marine', 0.9996452331542969), ('cart', 0.9996450543403625), ('ward', 0.9996443390846252), ('ryan', 0.9996442198753357), ('nex', 0.9996435642242432), ('dollar', 0.9996435046195984), ('pants', 0.9996432662010193), ('runner', 0.9996431469917297), ('bear', 0.999640941619873), ('wash', 0.999640941619873), ('them', 0.9996408224105835), ('handler', 0.9996406435966492), ('horse', 0.9996404647827148), ('iron', 0.9996404051780701), ('mag', 0.9996390342712402), ('court', 0.9996389746665955), ('andy', 0.9996383786201477), ('sight', 0.9996383190155029), ('bike', 0.9996379017829895), ('mic', 0.9996376037597656), ('will', 0.9996376037597656), ('her', 0.9996374845504761), ('wine', 0.999637246131897), ('ling', 0.9996371269226074), ('guns', 0.9996369481086731), ('walking', 0.999636709690094), ('owner', 0.9996363520622253), ('father', 0.9996362328529358), ('wide', 0.9996361136436462), ('oy', 0.9996361136436462), ('son', 0.999635636806488), ('graduate', 0.9996351599693298), ('town', 0.9996350407600403), ('cock', 0.9996350407600403), ('sky', 0.9996348023414612), ('leader', 0.9996346831321716), ('feeding', 0.9996342658996582), ('him', 0.9996336102485657), ('cow', 0.9996330142021179), ('lock', 0.9996329545974731), ('dy', 0.999632716178894), ('worn', 0.9996324181556702), ('ird', 0.9996322989463806), ('lay', 0.9996322393417358), ('bits', 0.9996309280395508), ('wal', 0.9996306896209717), ('parent', 0.9996305704116821), ('gal', 0.9996304512023926), ('member', 0.9996301531791687), ('hero', 0.9996299147605896), ('reed', 0.9996297359466553), ('ams', 0.9996296763420105), ('mit', 0.9996292591094971), ('rider', 0.9996291995048523), ('mud', 0.9996291399002075), ('hold', 0.9996287822723389), ('ebra', 0.9996285438537598), ('jan', 0.999628484249115), ('chair', 0.9996283650398254), ('god', 0.9996281862258911), ('edge', 0.9996276497840881), ('venue', 0.9996275305747986), ('mega', 0.9996274709701538), ('masters', 0.9996272921562195), ('master', 0.999626636505127), ('ork', 0.9996261596679688), ('owners', 0.9996255040168762), ('mine', 0.9996255040168762), ('mount', 0.9996251463890076), ('mate', 0.9996249079704285), ('witch', 0.9996247887611389), ('den', 0.9996243119239807)]
    # anchor_word: "person"
    # sim_dict: [('person', 1.0), ('people', 0.9997601509094238), ('guy', 0.999675452709198), ('professional', 0.9996753931045532), ('member', 0.999674916267395), ('monster', 0.9996553063392639), ('them', 0.9996500611305237), ('women', 0.9996476769447327), ('men', 0.9996448159217834), ('someone', 0.9996404051780701), ('human', 0.9996393918991089), ('personal', 0.9996389746665955), ('john', 0.9996382594108582), ('woman', 0.9996371865272522), ('doctor', 0.9996370673179626), ('wide', 0.9996362924575806), ('ana', 0.9996356964111328), ('figure', 0.9996354579925537), ('bits', 0.9996344447135925), ('president', 0.9996316432952881), ('record', 0.9996302127838135), ('bank', 0.9996272921562195), ('formation', 0.9996247887611389), ('pocket', 0.9996243119239807), ('reason', 0.9996233582496643), ('models', 0.99962317943573), ('position', 0.9996194243431091), ('central', 0.999618411064148), ('mag', 0.9996169209480286), ('keys', 0.9996165633201599), ('piece', 0.9996156692504883), ('man', 0.9996154308319092), ('smart', 0.9996152520179749), ('raised', 0.9996129274368286), ('crew', 0.9996117353439331), ('ghost', 0.9996114373207092), ('father', 0.9996111392974854), ('ump', 0.9996111392974854), ('graduate', 0.9996110200881958), ('cock', 0.999610960483551), ('highest', 0.999610960483551), ('boy', 0.9996100664138794), ('king', 0.9996100664138794), ('hero', 0.9996100664138794), ('fifth', 0.9996085166931152), ('option', 0.9996084570884705), ('period', 0.9996080994606018), ('banks', 0.999608039855957), ('unit', 0.9996079802513123), ('they', 0.9996077418327332), ('ee', 0.9996073246002197), ('winner', 0.9996068477630615), ('there', 0.9996065497398376), ('bike', 0.9996050000190735), ('kid', 0.9996044635772705), ('him', 0.9996044039726257), ('loop', 0.9996042251586914), ('kin', 0.9996041655540466), ('motion', 0.9996030926704407), ('let', 0.9996028542518616), ('bas', 0.9996027946472168), ('leader', 0.9996025562286377), ('marine', 0.9996024370193481), ('panic', 0.9996023774147034), ('hospital', 0.9996021389961243), ('continental', 0.9996021389961243), ('bare', 0.9996020793914795), ('vice', 0.9996020793914795), ('ket', 0.9996019005775452), ('trip', 0.9996018409729004), ('moving', 0.9996016621589661), ('thing', 0.9996011853218079), ('peak', 0.9996010065078735), ('individual', 0.9996009469032288), ('interesting', 0.9996005296707153), ('van', 0.9996004104614258), ('criminal', 0.9996000528335571), ('cash', 0.9995998740196228), ('walker', 0.999599814414978), ('zero', 0.9995995759963989), ('slice', 0.9995994567871094), ('plane', 0.9995993375778198), ('atomic', 0.9995993375778198), ('drops', 0.9995989203453064), ('handler', 0.9995988607406616), ('tiny', 0.9995988607406616), ('cap', 0.9995987415313721), ('rose', 0.9995987415313721), ('features', 0.9995986223220825), ('core', 0.9995985627174377), ('demon', 0.9995981454849243), ('cycle', 0.9995975494384766), ('vi', 0.999597430229187), ('whatever', 0.9995973706245422), ('mother', 0.9995966553688049), ('determination', 0.999596118927002), ('model', 0.999595582485199), ('ones', 0.9995955228805542), ('buck', 0.9995953440666199), ('planes', 0.9995951652526855)]
    # anchor_word: "people"
    # sim_dict: [('people', 1.0), ('person', 0.9997601509094238), ('they', 0.9997248649597168), ('professional', 0.999724805355072), ('men', 0.9997193217277527), ('women', 0.9997156858444214), ('things', 0.9997113347053528), ('someone', 0.9997072219848633), ('human', 0.9997057318687439), ('monster', 0.9997053742408752), ('guy', 0.9997011423110962), ('features', 0.9996955394744873), ('president', 0.9996934533119202), ('let', 0.9996922612190247), ('them', 0.9996922016143799), ('models', 0.9996887445449829), ('member', 0.999688446521759), ('leader', 0.9996876120567322), ('bits', 0.9996864795684814), ('everyone', 0.9996861219406128), ('smart', 0.9996839761734009), ('ryan', 0.9996825456619263), ('crew', 0.9996804594993591), ('woman', 0.9996784925460815), ('kids', 0.9996780753135681), ('handler', 0.9996780157089233), ('groups', 0.9996775984764099), ('father', 0.9996768236160278), ('him', 0.9996750354766846), ('living', 0.9996740221977234), ('owners', 0.99967360496521), ('humans', 0.9996734857559204), ('ones', 0.9996733069419861), ('fifth', 0.9996727108955383), ('mag', 0.9996724128723145), ('keys', 0.9996720552444458), ('hero', 0.9996719360351562), ('bare', 0.9996716976165771), ('runs', 0.9996711611747742), ('guns', 0.9996708631515503), ('rose', 0.9996702671051025), ('male', 0.9996696710586548), ('masters', 0.9996694326400757), ('we', 0.9996693134307861), ('runner', 0.999669075012207), ('lets', 0.9996677041053772), ('highest', 0.9996674060821533), ('units', 0.9996670484542847), ('buck', 0.9996666312217712), ('interesting', 0.9996660947799683), ('john', 0.9996659159660339), ('marine', 0.9996656775474548), ('fake', 0.9996654987335205), ('leaders', 0.9996654987335205), ('winner', 0.999664843082428), ('bank', 0.9996646642684937), ('individual', 0.9996641874313354), ('demon', 0.9996625185012817), ('foreign', 0.9996618032455444), ('wide', 0.9996610283851624), ('continental', 0.9996610283851624), ('gre', 0.9996610283851624), ('boy', 0.9996601939201355), ('tale', 0.9996592402458191), ('mother', 0.9996588230133057), ('assembly', 0.9996588230133057), ('raised', 0.999658465385437), ('387', 0.9996578693389893), ('pocket', 0.9996577501296997), ('ways', 0.9996575117111206), ('she', 0.9996572732925415), ('forestation', 0.9996570944786072), ('cycles', 0.999656617641449), ('mage', 0.999656617641449), ('style', 0.9996564984321594), ('burning', 0.9996563196182251), ('bath', 0.9996562600135803), ('police', 0.9996557831764221), ('uncle', 0.9996557831764221), ('clips', 0.9996554255485535), ('clone', 0.9996552467346191), ('lane', 0.9996550679206848), ('religious', 0.9996549487113953), ('planes', 0.9996548891067505), ('ware', 0.9996548295021057), ('bear', 0.9996547102928162), ('big', 0.9996546506881714), ('lease', 0.9996546506881714), ('jet', 0.999654233455658), ('porter', 0.999654233455658), ('levels', 0.9996541738510132), ('iron', 0.9996540546417236), ('players', 0.9996539950370789), ('cycle', 0.9996538758277893), ('shake', 0.9996537566184998), ('ork', 0.9996537566184998), ('restricted', 0.9996532201766968), ('central', 0.9996529221534729), ('laws', 0.9996525049209595), ('bay', 0.9996524453163147)]

if __name__=='__main__':
    # extract_textual_features_from_mymodel(text_encoder_type = "roberta-base")
    # extract_hico_verb_features()
    # extract_textual_features_automodel(text_encoder_type = "princeton-nlp/sup-simcse-roberta-base")
    check_similar_words('people')