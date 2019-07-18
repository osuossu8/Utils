


# ===========================================
# target 毎に分割した df を excel 形式で出力する.
# ===========================================

def save_target_excel(df):
    for i in tqdm(df.target.unique().tolist()):
        df_ = df[df.target==i]
        try:
          df_.to_excel(f'drive/My Drive/dir_name/target_{i}.xlsx', sheet_name='a')
        except:
          pass

# =======================================================================================
# 特定の dir にある以下の名称で保存された xlsx ファイルから target, label, data カラムを有する df を生成する.
# =======================================================================================

'''
['39_labelnameA.xlsx',
 '91_labelnameB.xlsx',
 '52_labelnameC.xlsx',
 '73_labelnameD.xlsx',
 '49_labelnameE.xlsx',
 '57_labelnameF.xlsx',
 '31_labelnameG.xlsx',
 '74_labelnameH.xlsx',
 '28_labelnameI.xlsx', ・・・]
'''

df_list = []
for i in range(len(os.listdir('drive/My Drive/dir_name/'))):
    file_name = os.listdir('drive/My Drive/dir_name/')[i]
    target_name = file_name.split("_")[0]
    target_label = file_name.split("_")[1].split(".")[0]
    # print(f"target = {target_name}")
    # print(f"label = {target_label}")
    file_path = os.path.join('drive/My Drive/dir_name/', file_name)
    each_df = pd.read_excel(file_path, sheet_name=0, header=0)
    each_df["target"] = target_name
    each_df["label"] = target_label
    # print(each_df.shape)

    # columnAで最頻のdataだけ抽出
    most_freq = each_df["columnA"].value_counts().keys()[0]
    each_df = each_df[each_df["columnA"]==most_freq]
    # print(each_df.shape)
    each_df = each_df[["data", "label", "target"]]
    each_df.columns = ["data", "label", "target"]
    # print(each_df.head())
    df_list.append(each_df)
    
df = pd.concat(df_list, sort=True).reset_index(drop=True)
print(df.shape)