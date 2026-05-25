#pragma once

void embedding_lookup(
    float* embedding_table,
    int* token_ids,
    float* output,
    int seq_len,
    int hidden_size
);
