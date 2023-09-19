# 处理多域数据：每个域读取数据，对每个item标记相应的域序号 -> 将所有域中的数据混合 -> 依据相同的user，对交互的序列进行排序，得到混合域的inter序列 -> 对id取index时，使用全域上的序号，然后记录id_inter和对应的domain_inter
# 对于多模态信息，使用metadata中的文本，经过bert获取embedding
# 突然想到，其实每个用户对相应的item的评价，才更应该作为第一手用户视角下的item信息
from tqdm import tqdm
import json

def read_ratings(domain_id, rating_path):
    user_ids = list()
    item_ids = list()
    timestamp = list()
    with open(rating_path) as fr:
        rating_lines = fr.readlines()
        for line in rating_lines:
            line_list = line.strip().split()
            user_ids.append(line_list[0])
            item_ids.append(str(domain_id) + '_' + line_list[1])
            timestamp.append(int(''.join(line_list[3].split('-'))))         # 时间戳是精确到日期的，因此可能会出现某个用户同一天购买多个的item的情况，序列可能出现不精确，有时间需要看一下这种case的数量多少
            
    return list(zip(user_ids, item_ids, timestamp))        

# 注意此处最好增加一个长度筛选
def generate_inter(inter_list, least_seq_len=2):
    print('generating inters over all domains...')
    user_item_dict = dict()
    for user_id, item_id, timestamp in inter_list:
        if user_id not in user_item_dict:
            user_item_dict[user_id] = [[item_id, timestamp]]
        else:
            user_item_dict[user_id].append([item_id, timestamp])
    print('sorting')
    for user_id in user_item_dict.keys():
        if len(user_item_dict[user_id]) < least_seq_len:
            user_item_dict.pop(user_id)
        user_item_dict[user_id] = sorted(user_item_dict[user_id], key=lambda x: x[1])
    return user_item_dict


def read_metadata(domain_id, domain_name, metadata_path):
    # 实际数据有可能会出现文本数据比较少，因为description可能为空，那么就只有标题
    # 返回含prompt的文本数据，存储在dict里
    domain_prompt = f'This is a {domain_name} item, and the following is the detailed infomation of it: '
    id_text_dict = dict()
    
    with open(metadata_path) as fr:
        metadata_lines = fr.readlines()
        for data_line in metadata_lines:
            data_dict = json.loads(data_line)
            item_id = str(domain_id) + '_' + data_dict['asin']
            item_title = data_dict['title']
            title_prompt = f'The item is named as {item_title}. '
            if 'categories' in data_dict and data_dict['categories']:
                categories = ','.join(data_dict['categories'])
                categories_prompt = f'The categories of the item are {categories}. '
            else:
                categories_prompt = ''

            if 'description' in data_dict and data_dict['description']:
                description = data_dict['description']
                description_prompt = f'The description of the item is {description}'
            else:
                description_prompt = ''
                
            item_text = domain_prompt + title_prompt + categories_prompt + description_prompt
            id_text_dict[item_id] = item_text
            
    return id_text_dict

def merge_and_reindex_items(dataset_name, user_item_dict):
    itemid_map = dict()
    iditem_map = dict()
    idx = 1
    for user_id in user_item_dict:
        for item in user_item_dict[user_id]:
            if item not in itemid_map:
                itemid_map[item] = idx
                iditem_map[id] = item
                idx += 1
                
            
        
        