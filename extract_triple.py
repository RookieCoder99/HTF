import xlwt
from utils import extract
def extract_triples(hbt_model,save_weights_path,path,author,subject_model,object_model, tokenizer, id2rel):
    workbook = xlwt.Workbook(encoding='utf-8')
    ws = workbook.add_sheet('sheet1', cell_overwrite_ok=True)
    ws.write(0, 0, "head")
    ws.write(0, 1, "tail")
    ws.write(0, 2, "relation")

    hbt_model.load_weights(save_weights_path)
    triples=extract(path, subject_model, object_model, tokenizer, id2rel)
    count=0

    triple_str=""
    for triple_list in triples:
        for triple in triple_list:
            count += 1
            ws.write(count, 0, triple[0])
            ws.write(count, 1, triple[1])
            ws.write(count, 2, triple[2])
    workbook.save(path+author+".xls")