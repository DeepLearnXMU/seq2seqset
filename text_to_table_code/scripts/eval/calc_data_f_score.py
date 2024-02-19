import argparse

import bert_score
import numpy as np
import tqdm
from sacrebleu import sentence_chrf

from table_utils import (
    extract_table_by_name,
    parse_text_to_table,
    is_empty_table,
)

bert_scorer = None
metric_cache = dict()  # cache some comparison operations


def parse_table_element_to_relation(table, i, j, row_header: bool, col_header: bool):
    assert row_header or col_header
    relation = []
    if row_header:
        assert j > 0
        relation.append(table[i][0])
    if col_header:
        assert i > 0
        relation.append(table[0][j])
    relation.append(table[i][j])
    return tuple(relation)


def parse_table_to_data(table, row_header: bool, col_header: bool):  # ret: row_headers, col_headers, relation tuples
    if is_empty_table(table, row_header, col_header):
        return set(), set(), set()

    assert row_header or col_header
    row_headers = list(table[:, 0]) if row_header else []
    col_headers = list(table[0, :]) if col_header else []
    if row_header and col_headers:
        row_headers = row_headers[1:]
        col_headers = col_headers[1:]

    row, col = table.shape
    relations = []
    for i in range(1 if col_header else 0, row):
        for j in range(1 if row_header else 0, col):
            if table[i][j] != "":
                relations.append(parse_table_element_to_relation(table, i, j, row_header, col_header))
    return set(row_headers), set(col_headers), set(relations)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp')
    parser.add_argument('tgt')
    parser.add_argument('--row-header', default=False, action="store_true")
    parser.add_argument('--col-header', default=False, action="store_true")
    parser.add_argument('--table-name', default=None)
    parser.add_argument('--metric', default='E', choices=['E', 'c', 'BS-scaled', ],
                        help="E: exact match\nc: chrf\nBS-scaled: re-scaled BERTScore")
    parser.add_argument('--src', default=None)
    args = parser.parse_args()
    assert args.row_header or args.col_header
    print("Args", args)
    return args


def calc_similarity_matrix(tgt_data, pred_data, metric):
    def calc_data_similarity(tgt, pred):
        if isinstance(tgt, tuple):
            ret = 1.0
            for tt, pp in zip(tgt, pred):
                ret *= calc_data_similarity(tt, pp)
            return ret

        if (tgt, pred) in metric_cache:
            return metric_cache[(tgt, pred)]

        if metric == 'E':
            ret = int(tgt == pred)
        elif metric == 'c':
            ret = sentence_chrf(pred, [tgt, ]).score / 100
        elif metric == 'BS-scaled':
            global bert_scorer
            if bert_scorer is None:
                bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
            ret = bert_scorer.score([pred, ], [tgt, ])[2].item()
            ret = max(ret, 0)
            ret = min(ret, 1)
        else:
            raise ValueError(f"Metric cannot be {metric}")

        metric_cache[(tgt, pred)] = ret
        return ret
    # if metric == 'BS-scaled':
    #     global bert_scorer
    #     if bert_scorer is None:
    #         bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    #     matrix = [(tgt, pred) for tgt in tgt_data for pred in pred_data]
    #     ret = bert_scorer.score([i[0] for i in matrix], [i[1] for i in matrix])[2]
    #     return ret.reshape([len(pred_data), len(tgt_data)]).numpy()
    # else:
    return np.array([[calc_data_similarity(tgt, pred) for pred in pred_data] for tgt in tgt_data], dtype=float)


def metrics_by_sim(tgt_data, pred_data, metric):
    sim = calc_similarity_matrix(tgt_data, pred_data, metric)  # (n_tgt, n_pred) matrix
    prec = np.mean(np.max(sim, axis=0))
    recall = np.mean(np.max(sim, axis=1))
    if prec + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1, sim.shape[0], sim.shape[1]


if __name__ == "__main__":
    args = parse_args()

    hyp_data = []
    with open(args.hyp) as f:
        for line in f:
            line = line.strip()
            if args.table_name is not None:
                line = extract_table_by_name(line, args.table_name)
            hyp_data.append(parse_text_to_table(line))
    tgt_data = []
    with open(args.tgt) as f:
        for line in f:
            line = line.strip()
            if args.table_name is not None:
                line = extract_table_by_name(line, args.table_name)
            tgt_data.append(parse_text_to_table(line))
    src_data = []
    if args.src is not None:
        with open(args.src) as f:
            for line in f:
                line = line.strip()
                src_data.append(line)
    row_header_precision = []
    row_header_recall = []
    row_header_f1 = []
    row_header_n_tgt = []
    row_header_n_pred = []
    col_header_precision = []
    col_header_recall = []
    col_header_f1 = []
    col_header_n_tgt = []
    col_header_n_pred = []
    relation_precision = []
    relation_recall = []
    relation_f1 = []
    relation_n_tgt = []
    relation_n_pred = []
    for hyp_table, tgt_table in tqdm.tqdm(zip(hyp_data, tgt_data), total=len(hyp_data)):
        if is_empty_table(tgt_table, args.row_header, args.col_header):
            pass
        elif hyp_table is None or is_empty_table(hyp_table, args.row_header, args.col_header):
            if args.row_header:
                row_header_precision.append(0)
                row_header_recall.append(0)
                row_header_f1.append(0)
            if args.col_header:
                # print(hyp_table)
                # print(tgt_table)
                col_header_precision.append(0)
                col_header_recall.append(0)
                col_header_f1.append(0)
            relation_precision.append(0)
            relation_recall.append(0)
            relation_f1.append(0)
        else:
            hyp_row_headers, hyp_col_headers, hyp_relations = parse_table_to_data(hyp_table, args.row_header,
                                                                                  args.col_header)
            tgt_row_headers, tgt_col_headers, tgt_relations = parse_table_to_data(tgt_table, args.row_header,
                                                                                  args.col_header)
            if args.row_header:
                p, r, f, n_tgt, n_pred = metrics_by_sim(tgt_row_headers, hyp_row_headers, args.metric)
                row_header_precision.append(p)
                row_header_recall.append(r)
                row_header_f1.append(f)
                row_header_n_tgt.append(n_tgt)
                row_header_n_pred.append(n_pred)
            if args.col_header:
                p, r, f, n_tgt, n_pred = metrics_by_sim(tgt_col_headers, hyp_col_headers, args.metric)
                col_header_precision.append(p)
                col_header_recall.append(r)
                col_header_f1.append(f)
                col_header_n_tgt.append(n_tgt)
                col_header_n_pred.append(n_pred)
            if len(hyp_relations) == 0:
                relation_precision.append(0.0)
                relation_recall.append(0.0)
                relation_f1.append(0.0)
                relation_n_tgt.append(0)
                relation_n_pred.append(0)
            else:
                p, r, f, n_tgt, n_pred = metrics_by_sim(tgt_relations, hyp_relations, args.metric)
                relation_precision.append(p)
                relation_recall.append(r)
                relation_f1.append(f)
                relation_n_tgt.append(n_tgt)
                relation_n_pred.append(n_pred)

    # print('Macro-averaged results:')
    result_dict = {}
    if args.row_header:
        result_dict['row_header'] = row_header_f1
        print("Row header: precision = %.2f; recall = %.2f; f1 = %.2f" % (
            np.mean(row_header_precision) * 100, np.mean(row_header_recall) * 100, np.mean(row_header_f1) * 100))
    if args.col_header:
        result_dict['col_header'] = col_header_f1
        print("Col header: precision = %.2f; recall = %.2f; f1 = %.2f" % (
            np.mean(col_header_precision) * 100, np.mean(col_header_recall) * 100, np.mean(col_header_f1) * 100))
    result_dict['relation'] = relation_f1
    print("Non-header cell: precision = %.2f; recall = %.2f; f1 = %.2f" % (
        np.mean(relation_precision) * 100, np.mean(relation_recall) * 100, np.mean(relation_f1) * 100))
    np.save(f'{args.hyp}.{args.metric}', result_dict)
    # print('Micro-averaged results:')
    # if args.row_header:
    #     p = np.average(row_header_precision, weights=row_header_n_pred)
    #     r = np.average(row_header_recall, weights=row_header_n_tgt)
    #     f = 2 * p * r / (p + r)
    #     print("Row header: precision = %.2f; recall = %.2f; f1 = %.2f" % (p * 100, r * 100, f * 100))
    # if args.col_header:
    #     p = np.average(col_header_precision, weights=col_header_n_pred)
    #     r = np.average(col_header_recall, weights=col_header_n_tgt)
    #     f = 2 * p * r / (p + r)
    #     print("Col header: precision = %.2f; recall = %.2f; f1 = %.2f" % (p * 100, r * 100, f * 100))
    # p = np.average(relation_precision, weights=relation_n_pred)
    # r = np.average(relation_recall, weights=relation_n_tgt)
    # f = 2 * p * r / (p + r)
    # print("Non-header cell: precision = %.2f; recall = %.2f; f1 = %.2f" % (p * 100, r * 100, f * 100))

            
