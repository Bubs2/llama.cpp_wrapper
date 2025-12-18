#pragma once

#include "id_chunk.h"
#include "chat.h"

namespace llama_server {
    class HistoryManager;
}

namespace llama_server::internal {

    struct ChatParamsAccessor {
        static common_chat_params& get_params(HistoryManager& hm);
    };

    struct ChunksAccessor {
        static std::vector<IDChunkPtr> get_chunks(HistoryManager& hm, size_t token_bound);
    };

}