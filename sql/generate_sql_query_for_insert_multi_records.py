import pandas as pd


def generate_query_for_insert_multi_records(df):
    """
    複数レコードの insert script を生成する.
    """
    query = "insert into proposal_page_label_manual \
             (created_at, updated_at, columnA, user_id, label_id, columnB) values "
    for columnA, columnB in zip(df.columnA.values, df.columnB.values):
        query1 = f"(now(), now(), {columnA}, 'kaerururu', 123, {columnB}), "
        query = query + query1
    return query    

df = pd.read_csv("/to/your/file/path")
# df.columns = ["columnA", "columnB"]

query = generate_query_for_insert_multi_records(df)
print(query)
