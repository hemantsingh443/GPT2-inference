#pragma once

void embedding_lookup(
    float* embedding_table,
    int* token_ids,
    float* output,
    int hidden_size
);
